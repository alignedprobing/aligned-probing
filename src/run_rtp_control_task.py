import os
import traceback
import click
@click.command()
@click.option('--model_name', type=str, default="allenai/OLMo-7B-hf")
@click.option('--chat_template_model_name', type=str, default="")
@click.option('--model_precision', type=str, default="full")
@click.option('--control_task', type=str, default="RANDOMIZATION")
@click.option('--template_index', type=str, default=0)
@click.option('--device', type=str, default="0")
@click.option('--subsets', type=str, default="input,forward,output,backward")
def main(
        model_name, chat_template_model_name, model_precision, control_task,
        template_index, device, subsets
):

    ## python3 probe_rtp.py --model_name allenai/OLMo-7B-hf --chat_template_model_name allenai/OLMo-7B-hf --model_precision halft --control_task NONE --template_index 0 --probe_type  linear  --device 0
    ## python3 probe_rtp.py --model_name allenai/OLMo-7B-Instruct-hf --chat_template_model_name allenai/OLMo-7B-Instruct-hf --model_precision full --control_task NONE --template_index 0 --probe_type  linear  --device 0

    subsets = subsets.split(",")

    if "input" in subsets:
        command = (
            f"python3 probe_rtp_input.py "
            f"--device {device} "
            f"--chat_template_model_name {chat_template_model_name} "
            f"--model_name {model_name} "
            f"--template_index {template_index} "
            f"--control_task {control_task} "
            f"--model_precision {model_precision} "
            f"--probe_type  linear "
            f"--probing_labels toxicity "
        )

        print(command)
        os.system(command)

    if "forward" in subsets:
        command = (
            f"python3 probe_rtp_forward.py "
            f"--device {device} "
            f"--chat_template_model_name {chat_template_model_name} "
            f"--model_name {model_name} "
            f"--template_index {template_index} "
            f"--control_task {control_task} "
            f"--model_precision {model_precision} "
            f"--probe_type  linear "
            f"--probing_labels toxicity "
        )

        print(command)
        os.system(command)

    if "backward" in subsets:
        command = (
            f"python3 probe_rtp_backward.py "
            f"--device {device} "
            f"--chat_template_model_name {chat_template_model_name} "
            f"--model_name {model_name} "
            f"--template_index {template_index} "
            f"--control_task {control_task} "
            f"--model_precision {model_precision} "
            f"--probe_type  linear "
            f"--probing_labels max_toxicity "
        )
    
        print(command)
        os.system(command)

    if "output" in subsets:
        command = (
            f"python3 probe_rtp_generation.py "
            f"--device {device} "
            f"--chat_template_model_name {chat_template_model_name} "
            f"--model_name {model_name} "
            f"--template_index {template_index} "
            f"--control_task {control_task} "
            f"--model_precision {model_precision} "
            f"--probe_type  linear "
            f"--probing_labels toxicity "
        )
    
        print(command)
        os.system(command)




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()