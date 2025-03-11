import itertools
import json
import os
import random
import sys
from typing import Union, List
import numpy
import click
import pandas
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import GenerationConfig

from encoders.OpenAssistantEncoder import OpenAssistantEncoder
from encoders.RealToxicPromptsEncoder import RealToxicPromptsEncoder
from encoders.TokenizerFix import char_to_token
from experiment_utils import get_encoding_path
from model_utils import extract_output_hidden_state, extract_input_hidden_state, load_model_tokenizer

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    rng = numpy.random.default_rng(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###
# todo
# - adopt for enc/dec
# - adopt for instructions
# - BERTScore
###
@click.command()
@click.option('--task', type=str, default="real_toxic_prompts")
@click.option('--model_name', type=str, default="allenai/OLMo-7B-hf")
@click.option('--tokenizer_model_name', type=str)
@click.option('--chat_template_model_name', type=str, default="")
@click.option('--model_architecture', type=str, default="decoder")
@click.option('--model_precision', type=str, default="full")
@click.option('--encoding_batch_size', type=int, default=4)
@click.option('--force_encoding', is_flag=True, default=False)
@click.option('--device', type=str, default="cpu")
@click.option('--template_index', type=str, default=0)
@click.option('--temperature', type=int, default=1)
@click.option('--num_return_sequences', type=int, default=25)
@click.option('--batch_split', type=int, default=1)
@click.option('--seed', type=int, default=0)
@click.option('--skip_layers', type=str)
@click.option('--start_generation_id', type=int, default=0)
def main(
        task, model_name, tokenizer_model_name, chat_template_model_name, model_architecture, model_precision, encoding_batch_size,
        force_encoding, device, template_index, temperature, num_return_sequences, batch_split, seed, skip_layers, start_generation_id
):



    encoding_folder = os.environ['ENCODING_PATH']

    seed_all(seed)
    if chat_template_model_name is None:
        chat_template_model_name = model_name

    template_indices = [int(index) for index in template_index.split(",")]

    for parsed_template_index in template_indices:

        encoding_folder_structure = get_encoding_path(encoding_folder, task, model_name, chat_template_model_name, model_precision, parsed_template_index, num_return_sequences, skip_layers=skip_layers)

        os.system(f"mkdir -p {encoding_folder_structure}")

        if os.path.exists(f"{encoding_folder_structure}/layer-0/input.feather") and not force_encoding:
            print("encodings already exists")
            continue


        model, tokenizer, chat_template_tokenizer = load_model_tokenizer(model_name, chat_template_model_name, model_architecture, model_precision, device, tokenizer_model_name=tokenizer_model_name, skip_layers=skip_layers)

        if skip_layers:
            model_name = f"{model_name}-wo-{skip_layers}"

        task_types = json.load(open("../defs/task_types.json"))
        samples = pandas.read_json(f"../tasks/{task}.jsonl", lines=True, orient="records")

        if "sub_task" in samples.columns:
            if os.path.exists(f"{encoding_folder_structure.replace(task, task + '-' + str(samples.sub_task.unique()[0]))}/layer-0/input.feather") and not force_encoding:
                print("encodings already exists")
                continue
        # dev
        if device == "cpu":
            samples = samples.sample(20, random_state=0)

        samples = samples[samples.notnull().all(1)]

        if "os_" in task:
            encoder = OpenAssistantEncoder()
        else:
            encoder = RealToxicPromptsEncoder()

        samples["instructions"] = samples.apply(lambda row: encoder.get_instructions(task_types=task_types, entry=row, template_index=parsed_template_index), axis=1)

        def apply_chat_template(instruction):
            if chat_template_tokenizer.chat_template:
                if "system" not in chat_template_tokenizer.chat_template or "System role not supported" in chat_template_tokenizer.chat_template:
                    system_prompt = instruction[0]["content"]
                    instruction[1]["content"] = f"{system_prompt}\n{instruction[1]['content']}"

                    instruction = instruction[1:]

                return chat_template_tokenizer.apply_chat_template(conversation=instruction, tokenize=False, add_generation_prompt=True)
            else:
                return " ".join([ele["content"] for ele in instruction]).strip()

        # compile instructions
        samples["compiled_instruction_text"] = samples["instructions"].apply(
            lambda instruction: apply_chat_template(instruction=instruction)
        )

        dataset = Dataset.from_pandas(samples)

        input_encodings = []
        instruction_encodings = []
        generated_encodings = []

        for batch in tqdm(dataset.iter(batch_size=encoding_batch_size)):
            batch_frame = pandas.DataFrame(batch)

            if batch_split == 1:
                final_num_return_sequences = num_return_sequences
            else:
                if int(num_return_sequences/batch_split) * batch_split != num_return_sequences:
                    print(f"batch split {batch_split} is not valid value")
                    sys.exit()
                final_num_return_sequences = int(num_return_sequences/batch_split)

            for i in range(batch_split):
                encoded_batch = tokenizer(batch["compiled_instruction_text"], padding=True, truncation=True, return_tensors='pt', return_offsets_mapping=True).to(device)

                #if "Llama-3".lower() in model_name.lower():
                #    encoded_batch.char_to_token = char_to_token

                model_encoded_batch = {
                    "input_ids": encoded_batch["input_ids"],
                    "attention_mask": encoded_batch["attention_mask"],
                }
                decoded_batch = tokenizer.batch_decode(encoded_batch["input_ids"], skip_special_tokens=True)

                generation_config = GenerationConfig(**{
                    'temperature': temperature,
                    'do_sample': True,
                    'num_return_sequences': final_num_return_sequences,
                    'num_beams': 1,
                    'max_new_tokens': 20,
                    'top_p': 0.9,
                    'top_k': 0,
                    'pad_token_id': tokenizer.eos_token_id,
                    'pad_token': tokenizer.eos_token,
                    'return_dict_in_generate': True,
                    'output_hidden_states': True,
                    'output_scores': True
                })

                with torch.no_grad():
                    generation = model.generate(
                        **model_encoded_batch,
                        generation_config=generation_config
                    )

                final_texts = tokenizer.batch_decode(generation.sequences, skip_special_tokens=True)

                decoded_batch = list(itertools.chain.from_iterable([[ele] *  final_num_return_sequences for ele in decoded_batch]))
                generated_texts = [
                    final_text.replace(decoded_input_text, "")
                    for final_text, decoded_input_text in zip(final_texts, decoded_batch)
                ]

                # (batch_size b, layer l, input_tokens i, embeddings e)
                input_hidden_states = extract_input_hidden_state(generation, final_num_return_sequences)

                # (batch_size b, layer l, input_tokens i, embeddings e)
                output_hidden_states = extract_output_hidden_state(generation)

                input_encoding, instruction_encoding = encoder.get_input_instruction_hidden_state(input_hidden_states, encoded_batch, batch_frame)
                generated_encoding = encoder.get_generated_hidden_state(output_hidden_states, encoded_batch, batch_frame, generated_texts, final_num_return_sequences)

                if batch_split > 1:
                    for ele in generated_encoding:
                        ele["generation_id"] = ele["generation_id"] + (i * 5)

                input_encodings.extend(input_encoding)
                instruction_encodings.extend(instruction_encoding)
                generated_encodings.extend(generated_encoding)

        input_encoding_frame = pandas.DataFrame(input_encodings).reset_index(drop=True)
        instruction_encoding_frame = pandas.DataFrame(instruction_encodings).reset_index(drop=True)
        generated_encoding_frame = pandas.DataFrame(generated_encodings).reset_index(drop=True)

        if batch_split > 1:
            input_encoding_frame = input_encoding_frame.drop_duplicates(["prompt_id", "layer"])

        for frame in [input_encoding_frame, instruction_encoding_frame, generated_encoding_frame]:
            del frame["instructions"]
            for col in frame.columns:
                if "level" in col:
                    del frame[col]

        if "sub_task" in samples.columns:
            for (sub_task, layer), layer_frame in input_encoding_frame.groupby(["sub_task", "layer"]):
                sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')

                os.system(f"mkdir -p {sub_task_folder}/layer-{layer}")
                relevant_columns = [col for col in layer_frame.columns if "generated_continuation" not in col]
                layer_frame[relevant_columns].reset_index().to_feather(f"{sub_task_folder}/layer-{layer}/input.feather")

            for (sub_task, layer), layer_frame in instruction_encoding_frame.groupby(["sub_task", "layer"]):
                sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')

                os.system(f"mkdir -p {sub_task_folder}/layer-{layer}")
                relevant_columns = [col for col in layer_frame.columns if "generated_continuation" not in col]
                layer_frame[relevant_columns].reset_index().to_feather(f"{sub_task_folder}/layer-{layer}/instruction.feather")

            for (sub_task, generation_id, layer), layer_frame in generated_encoding_frame.groupby(["sub_task", "generation_id", "layer"]):
                sub_task_folder = encoding_folder_structure.replace(task, f'{task}-{sub_task}')

                generation_id = start_generation_id + generation_id
                layer_frame["generation_id"] = generation_id
                os.system(f"mkdir -p {sub_task_folder}/layer-{layer}")
                relevant_columns = ["prompt_id", "prompt_text"] + [col for col in layer_frame.columns if not col.startswith("prompt_") ]
                layer_frame[relevant_columns].reset_index().to_feather(f"{sub_task_folder}/layer-{layer}/generation_{generation_id}.feather")

        else:
            for layer, layer_frame in input_encoding_frame.groupby("layer"):
                os.system(f"mkdir -p {encoding_folder_structure}/layer-{layer}")
                relevant_columns = [col for col in layer_frame.columns if "generated_continuation" not in col]
                layer_frame[relevant_columns].reset_index().to_feather(f"{encoding_folder_structure}/layer-{layer}/input.feather")

            for layer, layer_frame in instruction_encoding_frame.groupby("layer"):
                os.system(f"mkdir -p {encoding_folder_structure}/layer-{layer}")
                relevant_columns = [col for col in layer_frame.columns if "generated_continuation" not in col]
                layer_frame[relevant_columns].reset_index().to_feather(f"{encoding_folder_structure}/layer-{layer}/instruction.feather")

            for (generation_id, layer), layer_frame in generated_encoding_frame.groupby(["generation_id", "layer"]):
                generation_id = start_generation_id + generation_id
                layer_frame["generation_id"] = generation_id
                os.system(f"mkdir -p {encoding_folder_structure}/layer-{layer}")
                relevant_columns = ["prompt_id", "prompt_text"] + [col for col in layer_frame.columns if not col.startswith("prompt_") ]
                layer_frame[relevant_columns].reset_index().to_feather(f"{encoding_folder_structure}/layer-{layer}/generation_{generation_id}.feather")


if __name__ == "__main__":
    main()

