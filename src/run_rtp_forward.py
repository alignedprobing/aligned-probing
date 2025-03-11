import os
import traceback
import click
@click.command()
@click.option('--model_name', type=str, default="allenai/OLMo-7B-hf")
@click.option('--chat_template_model_name', type=str)
@click.option('--task', type=str, default="real_toxic_prompts")
@click.option('--model_precision', type=str, default="full")
@click.option('--control_task', type=str, default="NONE")
@click.option('--template_index', type=str, default=0)
@click.option('--probe_type', type=str, default="linear")
@click.option('--device', type=str, default="linear")
@click.option('--probing_labels', type=str, default="toxicity")
def main(
        model_name, chat_template_model_name, task, model_precision, control_task, template_index, probe_type, device, probing_labels
):

    for i in range(1):
        command = (
            f"CUDA_VISIBLE_DEVICES={device} python3 lift_probe_cross.py "
            f"--task {task} "
            f"--chat_template_model_name {chat_template_model_name} "
            f"--model_name {model_name} "
            f"--template_index {template_index} "
            f"--control_task {control_task} "
            f"--model_precision {model_precision} "
            f"--probe_type {probe_type} "
            f"--processing 'parallel'  "
            f"--probing_labels {probing_labels} "
            f"--num_return_sequences 25 "
            f"--in_filter generation_scored_{i}. "
            f"--cache_folder /ray"
        )

        print(command)
        os.system(command)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()