<div align="center">
<img style="vertical-align:middle" height="300" src="aligned_probing.svg" />
    <p>
        <b><a href="https://alignedprobing.github.io/"><b>Project Page</b></a></b>
    <p>
</div>


> **Abstract:** We introduce aligned probing, a novel interpretability framework that aligns the behavior of language models (LMs), based on their outputs, and their internal representations (internals). Using this framework, we examine over 20 OLMo, Llama, and Mistral models, bridging behavioral and internal perspectives for toxicity for the first time. Our results show that LMs strongly encode information about the toxicity level of inputs and subsequent outputs, particularly in lower layers. Focusing on how unique LMs differ offers both correlative and causal evidence that they generate less toxic output when strongly encoding information about the input toxicity. We also highlight the heterogeneity of toxicity, as model behavior and internals vary across unique attributes such as Threat. Finally, four case studies analyzing detoxification, multi-prompt evaluations, model quantization, and pre-training dynamics underline the practical impact of aligned probing with further concrete insights. Our findings contribute to a more holistic understanding of LMs, both within and beyond the context of toxicity.

If you encounter an issue, something is broken, or if you have further questions either [email us](alignedprobing@gmail.com) or open an issue here.




<p align="center">
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=01A88D">
    </a>
    <a href="https://github.com/alignedprobing/aligned-probing">
        <img alt="Opensource" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

## Setting up the environment
To use this repository, please ensure your setup meets the following requirements:
* Python version 3.10.
* Install all necessary packages using pip install -r requirements.txt`.
* If you want to load language models with `four_bit,` install `bitsandbytes.` If you have trouble installing it, use the version [`0.42.0`](https://github.com/TimDettmers/bitsandbytes/tree/0.42.0) and verify the installation with `python3 -m bitsandbytes`.


## Setting up redis for experiment tracking
This repository use a redis instance to track experiments. You can run one with the following docker-compose file and run the command `docker compose up`.

```
networks:
  tutorials:
    name: tutorials
    driver: bridge

services:
  redis:
    image: redis:latest
    ports:
    - 6379:6379
    networks:
        - tutorials
    volumes:
        - ../redis-data:/data
  redis-insight:
    image: redis/redisinsight:latest
    ports:
    - 5540:5540
    networks:
        - tutorials
```

## Update enviroment variables

Next, make sure you update the `.env` file with the following entries:
- `REDIS_SERVER`: IP address of the redis server for experiment tracking
- `REDIS_PORT`: Corresponding port of the redis server
- `PERSPECTIVE_API_KEY`: The key to the perspective API
- `ENCODING_PATH`: The path to store the encodings, make sure that there is enought space > 1TB
- `RESULT_PATH`: The path to store predictions and probing models
- `CACHE_PATH`: The path to store intermediate checkpoints and results


## Experiment steps

### 1. Get text continuations and internal encodings

In a first step, we let a model complete all the prompts from our subset of the `RealToxicPrompts` dataset. Therefore, run `python3 encode.py` and specific the following important arguments:

- `model_name`: Huggingface tag of the model, for the case study `Detoxification` use the corresponding detoxed version of models like `BatsResearch/llama2-7b-detox-qlora` or local pre-training checkpoints of OLMo for the case study `Pre-Training Dynamics`. Please following [online](https://github.com/allenai/OLMo/blob/main/docs/Checkpoints.md) to get these checkpoints.
- `model_precision`: Defines how you want to load the model, default `full` for most experiments or `four_bit` or `half` for model quantization ones, see case study `Model Quantization`.
- `encoding_batch_size`: Defines how many prompts you want to process within one batch, lower this value if you run out of GPU memory.
- `device`: Device to load the model, either `cpu` or `cuda`.
- `template_index`: The template index to use (index in the file `defs/task_types.json`), default `0` or `1-4` in case of the multi-prompt experiments (see case study `Multi-Prompt Evaluation`)

### 2. Retrieve perspective scores

Next, we retrieve the PERSPECTIVE API scores for the continuations by running `python3 attach_scores_recursive.py`.

### 3. Probing experiments

With the encodings and the scores from PERSPECTIVE API, we can run the aligned probing experiments.

To run the main experiments, we run `python3 run_rtp.py` with the following important arguments:
- `model_name`: The huggingface tag of the encoding models you used above.
- `model_precision`: The precision of the encoding model, default `full` for most experiments or `four_bit` or `half` for model quantization ones.
- `device`: The index of device, for example `0` for the first.
- `template_index`: The template index to use (index in the file `defs/task_types.json`), default `0` or `1-4` in case of the multi-prompt experiments.

Then we run `python3 run_rtp_mdl.py` and `python3 run_rtp_control_task.py` to verify the probing setup based on the subjectivity and compression measures, with the following parameters:
- `model_name`: The huggingface tag of the encoding model you used above.
- `device`: The index of device, for example `0` for the first .



### 4. Gathering behavioral results
To get the behavioral results run `python3 gather_behavioral_results.py`.

### 5. Gathering internal results
To get all results from probing experiments run `python3 gather_internal_results.py`.


## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 