from typing import List
import numpy as np
import numpy.typing as npt
import openai

openai.api_key_path = "/Users/jonathan/keys/dev_OPENAI_API_KEY.txt"  # type: ignore [fixme, github problem]


def simple_chunk(text: str, max_chunk_size: int, stride: int) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + max_chunk_size])
        i += stride

    return out


def get_raw_data():
    # return "This is a test of the emergency broadcast system. This is only a test."
    with open(
        "/examples/example_data/financial_report/10ks/10k-meta-plain-text-ascii.txt"
    ) as f_10k:
        # TODO: [P1] dont truncate
        return f_10k.read()[:2000]


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> npt.ArrayLike:
    # print("get_emb")
    text = text.replace("\n", " ")
    emb: npt.ArrayLike = openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]  # type: ignore [fixme]
    out: npt.ArrayLike = np.array(emb)  # type: ignore [fixme]
    return out
