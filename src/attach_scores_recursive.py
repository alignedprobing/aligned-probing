import sys
from collections import defaultdict

from experiment_utils import get_encoding_path
from parsing_utils import get_text_properties, get_properties_for_texts
from scoring import get_perspective_attributes, evaluate_fluency


import glob
import os
import click
import pandas
from tqdm import tqdm

@click.command()
@click.option('--filter', type=str, default="**")
@click.option('--force', type=bool, default=False)
def main(filter, force):
    perspective_api_key = os.environ["PERSPECTIVE_API_KEY"]
    encoding_folder = os.environ["ENCODING_PATH"]
    base_files = glob.glob(f"{encoding_folder}/{filter}/**/layer-0/generation_*", recursive=True)

    print(base_files)

    for base_file in base_files:

        if not force and "scored" in base_file:
            continue

        file_name = base_file.split("/")[-1]
        base_path = "/".join(base_file.split("/")[:-2])

        print("load", base_file)

        if ".feather" in base_file:
            base_samples = pandas.read_feather(base_file)
        else:
            base_samples = pandas.read_pickle(base_file)

        if "generated_continuation_text" not in base_samples.columns:
            continue

        generated_texts = base_samples["generated_continuation_text"]

        if "continuation_text" in base_samples:
            truth_texts = base_samples["continuation_text"]
        else:
            truth_texts = None

        if "generated_continuation_toxicity" in base_samples.columns:
            continue

        text_properties = get_properties_for_texts(generated_texts)
        perspective_scores = get_perspective_attributes(generated_texts, perspective_api_key)

        files = glob.glob(f"{base_path}/**/{file_name}", recursive=True)

        for file in tqdm(files):

            if ".feather" in file:
                frame = pandas.read_feather(file)
            else:
                frame = pandas.read_pickle(file)

            for score_name, score_values in text_properties.items():
                frame[f"generated_continuation_{score_name}"] = list(score_values)

            for score_name, score_values in perspective_scores.items():
                frame[f"generated_continuation_{score_name}"] = list(score_values)


            if "scored" not in file_name:
                new_file_name = file.replace(
                    file_name,
                    file_name.replace("generation", "generation_scored")
                )
            else:
                new_file_name = file_name

            if "pickle" in base_file:
                new_file_name = new_file_name.replace(".pickle", ".feather")

            assert "prompt_id" in frame.columns and "generation_id" in frame.columns

            frame = frame.reset_index()

            for column in frame.columns:
                if "level_" in column:
                    del frame[column]

            frame.to_feather(new_file_name)

            if new_file_name != file_name:
                os.system(f"rm {file}")

if __name__ == "__main__":
    main()