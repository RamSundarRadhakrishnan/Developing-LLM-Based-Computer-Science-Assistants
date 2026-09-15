"""
Evaluation of Granite Guardian 3.1 2B on irrelevancy and solution proximality detection task.
Models will be loaded in FP16 using llama.cpp
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from torch.nn.functional import softmax
import math
from openai import OpenAI
from pathlib import Path


guardian_client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

import math

def parse_llamacpp_output(completion, safe_token="No", risky_token="Yes"):
    choice = completion.choices[0]
    generated = choice.text.strip()

    if generated.lower() == risky_token.lower():
        label = risky_token
    elif generated.lower() == safe_token.lower():
        label = safe_token
    else:
        label = "Failed"

    logprobs = choice.logprobs
    if logprobs is None or not logprobs.top_logprobs:
        return label, None

    alternatives = logprobs.top_logprobs[0]

    safe_prob = 1e-50
    risky_prob = 1e-50

    for token, logprob in alternatives.items():
        token = token.strip().lower()

        if token == safe_token.lower():
            safe_prob += math.exp(logprob)
        elif token == risky_token.lower():
            risky_prob += math.exp(logprob)

    risk_probability = risky_prob / (safe_prob + risky_prob)

    return label, risk_probability

def load_split(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert set([
            "id",
            "assignment_context",
            "user_query",
            "assistant_response",
            LABEL_COL,
        ]).issubset(df.columns)
        return df


def create_prompt(risks, tokenizer, risk_name, messages):
    guardian_config = risks[risk_name]
    prompt = tokenizer.apply_chat_template(
        messages = messages,
        guardian_config = guardian_config,
        tokenizer = False,
        add_generation_prompt = True,
    )
    return prompt

def run_model(prompt):
    response = guardian_client.completions.create(
        model="granite_guardian",
        prompt = prompt,
        temperature = 0,
        max_tokens = 1,
        logprobs = 20
    )
    label, risk_probability