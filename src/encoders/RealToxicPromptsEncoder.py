import itertools
import json
import os
from string import Template
from nltk.translate.bleu_score import sentence_bleu
from googleapiclient import discovery

import numpy
import torch

from encoders.TaskEncoder import TaskEncoder
from encoders.TokenizerFix import char_to_token_wrapper


class RealToxicPromptsEncoder(TaskEncoder):
    def get_instructions(self, task_types, entry, demonstration = False, template_index=0):

        entry_task_type = entry["task_type"]
        template = task_types[entry_task_type]["templates"][template_index]


        prompt_text = entry["prompt_text"]

        instructions = []

        instructions.append({
            "role": "system",
            "content": template["system"]
        })

        if demonstration:
            #todo
            pass

        instructions.append({
            "role": "user",
            "content": template["user"].format(prompt_text=prompt_text)
        })

        return instructions

    def extract_average_input_hidden_states(self, hidden_states, attention_mask):
        return torch.stack([
            position_hidden_states * position_attention_mask
            for position_hidden_states, position_attention_mask in zip(hidden_states, attention_mask)
        ]).sum(dim=0)/attention_mask.sum()

    def extract_average_generated_hidden_states(self, hidden_states, attention_mask, generated_tokens, eos_token=0):
        input_length = len(attention_mask)
        end_of_text_tokens = (generated_tokens[input_length:] == eos_token).nonzero(as_tuple=True)[0]

        if len(end_of_text_tokens) == 0:
            last_token = len(hidden_states)
        else:
            last_token = end_of_text_tokens.max()

        return hidden_states[:last_token].mean(dim=0)

    def get_input_instruction_hidden_state(self, input_hidden_states, encoded_batch, batch_frame):
        input_encodings = []
        instruction_encodings = []

        for hidden_states, (i, batch_element) in zip(input_hidden_states, batch_frame.iterrows()):
            input_string = batch_element["prompt_text"].replace("\n", "")
            full_string = batch_element["compiled_instruction_text"]

            if input_string not in full_string:
                print(input_string)
                print(full_string)
            start_index = full_string.index(input_string)
            end_index = start_index + len(input_string)

            all_indices = (encoded_batch["attention_mask"][i] == 1).nonzero(as_tuple=True)[0].detach().cpu()

            input_relevant_indices = torch.tensor(sorted([
                token_index
                for token_index in set([char_to_token_wrapper(encoded_batch, i, char_index) for char_index in range(start_index, end_index + 1)])
                if token_index is not None
            ])).detach().cpu()

            instruction_relevant_indices = all_indices[~all_indices.unsqueeze(1).eq(input_relevant_indices).any(1)]

            input_hidden_state = hidden_states[:,input_relevant_indices].mean(dim=1).detach().cpu()
            instruction_hidden_state = hidden_states[:,instruction_relevant_indices].mean(dim=1).detach().cpu()

            for layer in range(input_hidden_state.shape[0]):
                batch_element["layer"] = layer

                input_batch_element = batch_element.copy()
                instruction_batch_element = batch_element.copy()

                input_batch_element["inputs_encoded"] = input_hidden_state[layer].numpy()
                instruction_batch_element["inputs_encoded"] = instruction_hidden_state[layer].numpy()

                input_encodings.append(input_batch_element)
                instruction_encodings.append(instruction_batch_element)

        return input_encodings, instruction_encodings

    def get_generated_hidden_state(
            self, output_hidden_states, encoded_batch, batch_frame,
            generated_texts, num_return_sequences
    ):
        generated_encodings = []

        perspective_scores = {}
        n_generated_tokens = output_hidden_states.shape[2]

        batch_size = batch_frame.shape[0]
        prompt_ids = list(itertools.chain.from_iterable([[prompt_id] *  num_return_sequences for prompt_id in batch_frame["prompt_id"]]))
        attention_masks = list(itertools.chain.from_iterable([[attention_mask] *  num_return_sequences for attention_mask in encoded_batch["attention_mask"]]))
        generation_ids = list(range(num_return_sequences)) * batch_frame.shape[0]

        for hidden_state, generated_text, instance_attention_mask, prompt_id, generation_id in zip(output_hidden_states, generated_texts, attention_masks , prompt_ids, generation_ids):
            generation_attention_mask = instance_attention_mask[-n_generated_tokens:]
            n_relevant_tokens = generation_attention_mask.sum()
            batch_element = batch_frame.query(f"prompt_id=={prompt_id}").iloc[0]
            generation_hidden_state = hidden_state[:,:n_relevant_tokens].mean(dim=1).detach().cpu()

            for layer, layer_state in enumerate(generation_hidden_state):
                layer_batch_element = batch_element.copy()

                layer_batch_element["prompt_id"] = prompt_id
                layer_batch_element["generation_id"] = generation_id
                layer_batch_element["layer"] = layer
                layer_batch_element["inputs_encoded"] = layer_state.numpy()
                layer_batch_element["generated_continuation_text"] = generated_text

                scores = {} #self.evaluate_generated_text(generated_text, batch_element["continuation_text"])

                for name, score in scores.items():
                    layer_batch_element[f"continuation_{name}"] = score

                for name, score in perspective_scores.items():
                    layer_batch_element[f"generated_continuation_{name}"] = score

                generated_encodings.append(layer_batch_element)

        return generated_encodings

    def get_perspective_attribute(self, text):
        API_KEY = os.environ["PERSPECTIVE_API_KEY"]

        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        analyze_request = {
            'comment': { 'text': text },
            'requestedAttributes': {
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {},
                'TOXICITY': {},
                'SEXUALLY_EXPLICIT': {},
            }
        }
        try:
            response = client.comments().analyze(body=analyze_request).execute()

            return {
                score_name.lower(): score["summaryScore"]["value"]
                for score_name, score in response["attributeScores"].items()
            }

        except:
            return {}