import math
import numpy
from multiprocessing import Pool

import torch
import nltk
import pandas
from bert_score import score
from googleapiclient import discovery
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from retry import retry
from bert_score import BERTScorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

rouge = Rouge()

device = "cuda" if torch.cuda.is_available() else "cpu"


@retry(delay=0.2, backoff=1.2)
def execute_request(analyze_request, client, all_scores):

    try:
        response = client.comments().analyze(body=analyze_request).execute()
        return {
            score_name.lower(): score["summaryScore"]["value"]
            for score_name, score in response["attributeScores"].items()
        }
    except Exception as e:
        if "does not support request languages" in e.reason:
            print("unsupported language")
            return {
                score_name.lower(): -999
                for score_name in all_scores
            }
        elif "empty" in e.reason:
            print("empty")
            return {
                score_name.lower(): -999
                for score_name in all_scores
            }
        else:
            print(e.response.status_code)
            raise e
def call_perspective_api(params):

    i, text, client, language, attributes = params


    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {
            attribute: {}
            for attribute in attributes
        },
        'languages': [language]
    }
    print("run", i)
    return execute_request(analyze_request, client, attributes)

def get_perspective_attributes(texts, api_key, languages=[], pool=None):

    if pool is None:
        pool = Pool(24)

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )


    if len(languages) > 0:
        attributes = ['IDENTITY_ATTACK','INSULT','PROFANITY','THREAT','TOXICITY']
        params = [(i, text, client, language, attributes) for i, (text, language) in enumerate(zip(texts, languages))]
    else:
        attributes = ['IDENTITY_ATTACK','INSULT','PROFANITY','THREAT','TOXICITY','SEXUALLY_EXPLICIT']
        params = [(i, text, client, "en", attributes) for i, text in enumerate(texts)]


    scores = pool.map(call_perspective_api, params)

    return pandas.DataFrame(scores).to_dict(orient="list")