from functools import partial
import sys

import numpy as np
import openai
from semantic_retrieval.examples.report_10k.ingest import get_embedding
from sklearn.metrics.pairwise import cosine_similarity


def knn(query, candidates, metric_name, k):
    query_emb = np.expand_dims(get_embedding(query), axis=0)
    if metric_name == "cos_sim":
        sim_array = np.squeeze(cosine_similarity(query_emb, candidates))
        print(f"{sim_array.shape=}")
        # print(f"{sim_array=}")
        k = min(k, candidates.shape[0])
        arg_sorted = np.argsort(sim_array)
        # print(f"{arg_sorted=}")
        # print(f"{k=}, {arg_part.shape=}")
        top_idxs = arg_sorted[-k:]
        print(f"{top_idxs=}")

        # closest first
        return reversed(top_idxs)
    else:
        raise NotImplementedError


def generate_with_context(query, chunks):
    # TODO implement
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Who won the world series in 2020?"},
    #         {
    #             "role": "assistant",
    #             "content": "The Los Angeles Dodgers won the World Series in 2020.",
    #         },
    #         {"role": "user", "content": "Where was it played?"},
    #     ],
    # )

    # return response

    # print(f"{response=}")
    pass
