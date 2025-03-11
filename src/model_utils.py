import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from transformers.generation import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput

from adaptions.olmo import OlmoBypassDecoderLayer, Olmo2BypassDecoderLayer, LlamaBypassDecoderLayer, \
    MistralBypassDecoderLayer
from encoders.TokenizerFix import CustomLlama3Tokenizer
from torch.nn.modules.container import ModuleList

def load_model_tokenizer(model_name, chat_template_model_name, model_architecture, model_precision, device, tokenizer_model_name=None, skip_layers=None):
    if tokenizer_model_name is None:
        tokenizer_model_name = model_name

    if model_architecture == "decoder":
        #if "Llama-3".lower() in model_name.lower():
        #    tokenizer = CustomLlama3Tokenizer("meta-llama/Meta-Llama-3.1-8B-Instruct")
        #else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, padding_side="left")
        chat_template_tokenizer = AutoTokenizer.from_pretrained(chat_template_model_name, padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        chat_template_tokenizer = AutoTokenizer.from_pretrained(chat_template_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        chat_template_tokenizer.pad_token = chat_template_tokenizer.eos_token
        chat_template_tokenizer.pad_token_id = chat_template_tokenizer.eos_token_id

    if device == "cpu":
        if model_architecture == "decoder":
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        elif model_architecture == "encoder-decoder":
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        else:
            raise "Unsupported architecture"

    else:
        args = {
            "device_map": "auto",
            "trust_remote_code": True
        }
        if model_precision == "half":
            args["torch_dtype"] = torch.float16

        if model_precision == "four_bit":
            args["load_in_4bit"] = True

        if model_architecture == "decoder":
            model = AutoModelForCausalLM.from_pretrained(model_name,**args)
        elif model_architecture == "encoder-decoder":
            model = T5ForConditionalGeneration.from_pretrained(model_name, **args)
        else:
            raise "Unsupported architecture"

    if skip_layers:
        skip_layer_list = [int(layer_id) for layer_id in skip_layers.split(",")]
        if hasattr(model, "gpt_neox"):
            model.gpt_neox.layers = ModuleList([layer for i, layer in enumerate(model.gpt_neox.layers) if i not in skip_layer_list])

        elif hasattr(model, "model") and "olmo-2" in model_name.lower():
            model.model.layers = ModuleList([
                layer if i not in skip_layer_list else Olmo2BypassDecoderLayer(layer=layer)
                for i, layer in enumerate(model.model.layers)
            ])

        elif hasattr(model, "model") and "olmo" in model_name.lower() and "olmo-2" not in model_name.lower():
            model.model.layers = ModuleList([
                layer if i not in skip_layer_list else OlmoBypassDecoderLayer(layer=layer)
                for i, layer in enumerate(model.model.layers)
            ])
        elif hasattr(model, "model") and "llama" in model_name.lower():
            model.model.layers = ModuleList([
                layer if i not in skip_layer_list else LlamaBypassDecoderLayer(layer=layer)
                for i, layer in enumerate(model.model.layers)
            ])
        elif hasattr(model, "model") and "mistral" in model_name.lower():
            model.model.layers = ModuleList([
                layer if i not in skip_layer_list else MistralBypassDecoderLayer(layer=layer)
                for i, layer in enumerate(model.model.layers)
            ])
        else:
            raise "skip layer is not implemented for this model"
    return model, tokenizer, chat_template_tokenizer

def extract_input_hidden_state(generation, num_return_sequences):
    # (batch_size b, layer l, input_tokens i, embeddings e)

    if type(generation) == GenerateEncoderDecoderOutput or type(generation) == GenerateBeamEncoderDecoderOutput:
        input_hidden_states = torch.stack(generation.encoder_hidden_states).transpose(0,1)
    elif type(generation) == GenerateDecoderOnlyOutput:
        input_hidden_states = torch.stack(generation.hidden_states[0])[:,::num_return_sequences].transpose(0,1)
    elif type(generation) == GenerateBeamDecoderOnlyOutput:
        input_hidden_states = torch.stack(generation.hidden_states[0])[:,::num_return_sequences].transpose(0,1)
    else:
        raise "Unsupported Output"

    return input_hidden_states

def extract_output_hidden_state(generation):
    # (batch_size b, layer l, input_tokens i, embeddings e)

    if type(generation) == GenerateEncoderDecoderOutput or type(generation) == GenerateBeamEncoderDecoderOutput:
        output_hidden_states = torch.stack([torch.stack(layer_hidden_states) for layer_hidden_states in generation.decoder_hidden_states])
        output_hidden_states = output_hidden_states.transpose(0,3)[0].transpose(0,1)
    elif type(generation) == GenerateDecoderOnlyOutput or type(generation) == GenerateBeamDecoderOnlyOutput:
        output_hidden_states = torch.stack([torch.stack(layer_hidden_states) for layer_hidden_states in generation.hidden_states[1:]])
        output_hidden_states = output_hidden_states.transpose(0,3)[0].transpose(0,1)
    else:
        raise "Unsupported Output"

    return output_hidden_states