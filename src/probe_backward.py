import random
import signal
import string
import sys
from multiprocessing import Pool

import numpy

from experiment_utils import get_encoding_path, check_redis_path

sys.path.append("../probing/src")

import glob
import os
import traceback
from typing import Dict

import click
import pandas
import ray
import torch
from retry import retry

from itertools import product

from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam

from defs.control_task_types import CONTROL_TASK_TYPES
from defs.probe_task_types import PROBE_TASK_TYPES
from utils.data_loading import load_dataset
from utils.session_utils import run_probe_with_params, ray_run_probe_with_params, run_probe_with_params_pool
from logging import info


def scale_labels(labels):
    return (labels - labels.min()) / (labels.max() - labels.min())

def get_hyperparameters(hyperparameters:Dict):
    params = [dict(zip(hyperparameters, v)) for v in product(*hyperparameters.values())]
    params = [
        param
        for param in params
        if not (param["num_hidden_layers"] == 0 and param["hidden_dim"] > 0) and not (param["num_hidden_layers"] > 0 and param["hidden_dim"] == 0)
    ]
    return params

def get_session_id():
    return ''.join(random.choices(string.ascii_lowercase +
                                  string.digits, k=10))

def clean_session(sig=None, frame=None):
    print(f"clean {os.environ['CACHE_FOLDER']}")
    os.system("rm -rf " + os.environ["CACHE_FOLDER"])
    sys.exit(0)


CONFIG = {
    "probes_samples_path": "",
    "probe_task_type": PROBE_TASK_TYPES.SENTENCE,
    "num_probe_folds": 4,
    "num_labels": 2,
    "model_name": "bert-base-uncased",
    "output_dim": 768,
    "hyperparameters": {
        "learning_rate": [0.001],
        "batch_size": [16],
        "optimizer": [Adam],
        "hidden_dim": [0],
        "dropout": [0.2],
        "warmup_rate": [0.1],
        "num_hidden_layers": [0]
    }
}

@click.command()
@click.option('--task', type=str, default="real_toxic_prompts")
@click.option('--model_name', type=str, default="allenai/OLMo-7B-hf")
@click.option('--chat_template_model_name', type=str)
@click.option('--model_precision', type=str, default="full")
@click.option('--batch_size', type=int, default=16)
@click.option('--project_prefix', type=str, default="rtp")
@click.option('--seeds', type=str, default="0,1,2,3,4")
@click.option('--dump_preds', is_flag=True, default=True)
@click.option('--force_probing', is_flag=True, default=False)
@click.option('--probing_labels', type=str, default="toxicity")
@click.option('--logging', type=str, default="redis")
@click.option('--processing', type=str, default="seq")
@click.option('--control_task', type=str, default="NONE")
@click.option('--template_index', type=str, default=0)
@click.option('--num_return_sequences', type=int, default=25)
@click.option('--in_filter', type=str, default="input")
@click.option('--probe_type', type=str, default="linear")
@click.option('--limit', type=int)
def main(
        task, model_name, chat_template_model_name, model_precision, batch_size, project_prefix,
        seeds, dump_preds, force_probing, probing_labels,
        logging, processing, control_task, template_index, num_return_sequences, in_filter, probe_type, limit
):

    encoding_folder = os.environ['ENCODING_PATH']
    result_folder = os.environ['RESULT_PATH']
    cache_folder = os.environ['CACHE_PATH']

    project_prefix = project_prefix + "-back"

    if chat_template_model_name is None:
        chat_template_model_name = model_name

    model_name = model_name.replace('/', '_')
    chat_template_model_name = chat_template_model_name.replace('/', '_')

    control_task = CONTROL_TASK_TYPES[control_task]
    probing_labels = probing_labels.split(",")
    result_folder = os.path.abspath(result_folder)
    session_id = get_session_id()
    cache_folder = os.path.abspath(f"{cache_folder}/{session_id}")
    os.environ["CACHE_FOLDER"] = cache_folder

    os.system(f"mkdir -p {cache_folder}")

    CONFIG["hyperparameters"]["seed"] = [int(seed) for seed in seeds.split(",")]
    CONFIG["control_task_type"] = control_task
    CONFIG["probe_type"] = probe_type
    CONFIG["encoding"] = model_precision
    CONFIG["num_labels"] = 1
    CONFIG["model_name"] = model_name

    template_indices = [int(index) for index in template_index.split(",")]


    for parsed_template_index in template_indices:


        encoding_folder_structure = get_encoding_path(encoding_folder, task, model_name, chat_template_model_name, model_precision, parsed_template_index, num_return_sequences)

        torch.set_num_threads(1)

        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        encoding_files = glob.glob(f"{encoding_folder_structure}/**/input_reverse.*")
        print(encoding_folder_structure)
        print(encoding_files)
        for encoding_file in encoding_files:

            origin = "input" if "input" in encoding_file else "generation"
            file_name = encoding_file.split("/")[-1]
            if in_filter != None and in_filter not in encoding_file:
                continue

            if "feather" not in encoding_file and "pickle" not in encoding_file:
                continue

            layer_name = encoding_file.split("/")[-2]
            layer_id = int(layer_name.split("-")[1])

            if "feather" in encoding_file:
                frame = pandas.read_feather(encoding_file)
            else:
                frame = pandas.read_pickle(encoding_file)

            if limit is not None:
                print(f"probe on a subset of {limit} instances")
                frame = frame.head(limit)

            CONFIG["sample_size"] = frame.shape[0]

            default_params = []

            for probing_label in probing_labels:

                if origin == "input":
                    relevant_label = f"prompt_{probing_label}"
                    relevant_columns = ["prompt_id", "prompt_text", "inputs_encoded", relevant_label]
                    if relevant_label not in frame.columns:
                        continue
                    probe_name = f"{task}-prompt_{probing_label}"
                    probing_frame = frame[relevant_columns]
                    probing_frame.columns = ["prompt_id", "inputs", "inputs_encoded", "label"]
                    generation_id = -1
                elif origin == "generation":
                    relevant_label = f"generated_continuation_{probing_label}"
                    if relevant_label not in frame.columns:
                        relevant_label = probing_label
                        probe_name = f"{task}-generated_continuation_{probing_label}"
                    else:
                        probe_name = f"{task}-{relevant_label}"

                    if relevant_label not in frame.columns:
                        continue

                    relevant_columns = ["prompt_id", "generation_id", "generated_continuation_text", "inputs_encoded", relevant_label]

                    probing_frame = frame[relevant_columns]
                    probing_frame.columns = ["prompt_id", "generation_id", "inputs", "inputs_encoded", "label"]
                    probing_frame = probing_frame[probing_frame["label"] != -999]
                    probing_frame = probing_frame[probing_frame["label"] == probing_frame["label"]]
                    generation_id = list(probing_frame["generation_id"].unique())[0]
                else:
                    continue

                probing_frame = probing_frame[
                    [(ele != ele).sum() == 0 for ele in list(probing_frame["inputs_encoded"])]
                ]

                probing_frame["label"] = scale_labels(probing_frame["label"])

                if control_task == CONTROL_TASK_TYPES.RANDOMIZATION:
                    random_labels = numpy.random.permutation(probing_frame["label"].values)
                    probing_frame["label"] = random_labels

                kfold = KFold(n_splits=4, shuffle=True, random_state=42)

                frames = []

                if origin == "input":

                    for fold, (train_index, test_index) in enumerate(kfold.split(probing_frame)):
                        train_frame = probing_frame.iloc[train_index]
                        train_frame, dev_frame = train_test_split(train_frame, train_size=0.8, random_state=fold)
                        test_frame = probing_frame.iloc[test_index]

                        frames.append((-1, train_frame, dev_frame, test_frame, fold))

                elif origin == "generation":
                    for generation_id in probing_frame["generation_id"].unique():
                        sub_frame = probing_frame[probing_frame["generation_id"] == generation_id]

                        for fold, (train_index, test_index) in enumerate(kfold.split(sub_frame)):
                            train_frame = sub_frame.iloc[train_index]
                            train_frame, dev_frame = train_test_split(train_frame, train_size=0.8, random_state=fold)
                            test_frame = sub_frame.iloc[test_index]

                            frames.append((generation_id, train_frame, dev_frame, test_frame, fold))
                else:
                    continue


                checkup_project = f"{project_prefix}-{task}-{relevant_label}"

                layer_check_config = {
                    "project": checkup_project,
                    "model_name": model_name,
                    "chat_template_model_name": chat_template_model_name,
                    "model_precision": model_precision,
                    "control_task_type": control_task.name,
                    "probe_type": probe_type,
                    "generation_id": "generation-id:" + str(generation_id),
                    "template_index": "template:" + str(int(template_index)),
                    "sample_size": "sample_size:" + str(probing_frame.shape[0]),
                    "origin": origin,
                    "layer_id": "layer:" + str(layer_id),
                    #"seed": "seed-" + str(seed),
                    #"fold": "fold-" + str(fold),
                }

                if check_redis_path(layer_check_config) >= 20:
                    print(probing_label, "Layer Done")
                    continue

                for generation_id, train_frame, dev_frame, test_frame, fold in frames:
                    train_dataset = load_dataset(train_frame)
                    dev_dataset = load_dataset(dev_frame)
                    test_dataset = load_dataset(test_frame)

                    input_dim = probing_frame["inputs_encoded"].iloc[0].shape[-1]

                    for hyperparameter in get_hyperparameters(CONFIG["hyperparameters"]):

                        hyperparameter["fold"] = fold
                        hyperparameter["encoding_file"] = encoding_file
                        hyperparameter["origin"] = origin
                        hyperparameter["generation_id"] = generation_id
                        hyperparameter["layer_name"] = layer_name
                        hyperparameter["layer_id"] = layer_id
                        hyperparameter["num_labels"] = 1
                        hyperparameter["model_name"] = model_name
                        hyperparameter["template_index"] = parsed_template_index
                        hyperparameter["chat_template_model_name"] = chat_template_model_name
                        hyperparameter["control_task_type"] = CONFIG["control_task_type"].name
                        hyperparameter["probe_task_type"] = CONFIG["probe_task_type"].name
                        hyperparameter["encoding"] = CONFIG["encoding"]
                        hyperparameter["sample_size"] = probing_frame.shape[0]
                        hyperparameter["probe_type"] = CONFIG["probe_type"]

                        hyperparameter["redis_server"] = os.environ["REDIS_SERVER"]
                        hyperparameter["redis_port"] = os.environ["REDIS_PORT"]

                        hyperparameter["redis_run_fields"] = {
                            "project": checkup_project,
                            "model_name": model_name,
                            "chat_template_model_name": chat_template_model_name,
                            "model_precision": model_precision,
                            "control_task_type": control_task.name,
                            "probe_type": probe_type,
                            "generation_id": "generation-id:" + str(generation_id),
                            "template_index": "template:" + str(int(template_index)),
                            "sample_size": "sample_size:" + str(probing_frame.shape[0]),
                            "origin": origin,
                            "layer_id": "layer:" + str(layer_id),
                            "seed": "seed:" + str(hyperparameter["seed"]),
                            "fold": "fold:" + str(fold),
                        }

                        if check_redis_path(hyperparameter["redis_run_fields"]) >= 1:
                            print("Run done")
                            continue

                        param_ele = {
                            "hyperparameter": hyperparameter,
                            "input_dim": input_dim,
                            "n_layers": 1,
                            "result_folder": result_folder,
                            "cache_folder": cache_folder,
                            #**CONFIG
                        }
                        default_params.append((param_ele, train_dataset, dev_dataset, test_dataset, dump_preds, force_probing, project_prefix, logging, probe_name))

            if processing == "seq":
                for param in default_params:
                    run_probe_with_params(*param)

            if processing == "parallel":
                pool = Pool(4)
                pool.map(run_probe_with_params_pool, default_params)

    os.system(f"rm -rf {cache_folder}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    signal.signal(signal.SIGINT, clean_session)

    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()