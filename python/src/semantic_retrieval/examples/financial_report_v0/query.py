from typing import List

import numpy as np
import openai

import semantic_retrieval.common.types as types
from .ingest import get_embedding


from sklearn.metrics.pairwise import cosine_similarity  # type: ignore [fixme]


def knn(query: str, candidates: types.NPA, metric_name: str, k: int) -> List[int]:
    query_emb = np.expand_dims(get_embedding(query), axis=0)
    if metric_name == "cos_sim":
        cos_sim = cosine_similarity(query_emb, candidates)  # type: ignore [fixme]
        sim_array: types.NPA = np.squeeze(cos_sim)
        print(f"{sim_array.shape=}")
        # print(f"{sim_array=}")
        k = min(k, candidates.shape[0])
        arg_sorted = np.argsort(sim_array)
        # print(f"{arg_sorted=}")
        # print(f"{k=}, {arg_part.shape=}")
        top_idxs = arg_sorted[-k:]
        print(f"{top_idxs=}")

        # closest first
        return list(reversed(top_idxs))
    else:
        raise NotImplementedError


def generate_with_context(query: str, chunks: List[str]) -> str:
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Rearrange the context to answer the question. "
            "Do not include Any words that do not appear in the context. "
            "CONTEXT:"
        )
        + "\n * ".join(chunks)
        + "QUESTION:\n",
    }
    response = openai.ChatCompletion.create(  # type: ignore [fixme]
        model="gpt-4",
        messages=[
            system,
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0]["message"]["content"]  # type: ignore [fixme]
