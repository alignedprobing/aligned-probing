import os
import traceback
import click
@click.command()
@click.option('--model_name', type=str, default="allenai/OLMo-7B-hf")
@click.option('--chat_template_model_name', type=str, default="")
@click.option('--model_precision', type=str, default="full")
@click.option('--control_task', type=str, default="NONE")
@click.option('--template_index', type=str, default=0)
@click.option('--device', type=str, default="0")
@click.option('--subsets', type=str, default="input,forward,output,backward")
@click.option('--parallel', type=bool, default=False)
def main(
        model_name, chat_template_model_name, model_precision, control_task, template_index, device, subsets, parallel
):

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
            f"--probe_type linear_mdl "
            f"--probing_labels toxicity "
        )

        if parallel:
            command = command + " &"
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
            f"--probe_type linear_mdl "
            f"--probing_labels toxicity "
        )

        if parallel:
            command = command + " &"
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
            f"--probe_type linear_mdl "
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
            f"--probe_type linear_mdl "
            f"--probing_labels toxicity "
        )

        if parallel:
            command = command + " &"
        print(command)
        os.system(command)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()