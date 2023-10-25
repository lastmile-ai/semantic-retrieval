import sys
import numpy as np
import openai

openai.api_key_path = "/Users/jonathan/keys/dev_OPENAI_API_KEY.txt"


def simple_chunk(text, max_chunk_size, stride):
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + max_chunk_size])
        i += stride

    return out


def get_raw_data():
    return "This is a test of the emergency broadcast system. This is only a test."
    with open("../examples/example_data/10k/10k-meta-plain-text-ascii.txt") as f_10k:
        print(f_10k.read())


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return np.array(
        openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
    )
