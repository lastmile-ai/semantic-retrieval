import json
import sys
import numpy as np
from semantic_retrieval.examples.report_10k.ingest import (
    get_embedding,
    get_raw_data,
    simple_chunk,
)
from semantic_retrieval.examples.report_10k.query import generate_with_context, knn


def main(argv):
    data = get_raw_data()
    print(f"{data=}")
    max_chunk_size = 10
    stride = 1
    chunked = simple_chunk(data, max_chunk_size, stride)
    print(f"chunked={json.dumps(list(enumerate(chunked)), indent=2)}")
    embs = np.stack([get_embedding(chunk) for chunk in chunked])

    queries = argv[1].split(",")
    for query in queries:
        print(f"\n\n{query=}")
        k = 5
        top_k_idxs = knn(query, embs, "cos_sim", k)
        top_k = [chunked[i] for i in top_k_idxs]

        print(f"{top_k=}")

    # resp = generate_with_context(query, top_k)
    # print(f"{resp=}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
