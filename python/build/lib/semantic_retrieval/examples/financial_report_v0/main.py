import sys
import numpy as np
from semantic_retrieval.common import types
from semantic_retrieval.common.types import Record

# from semantic_retrieval.common import types

from semantic_retrieval.examples.financial_report_v0.ingest import (
    get_embedding,
    get_raw_data,
    simple_chunk,
)


from pydantic.dataclasses import dataclass

from semantic_retrieval.examples.financial_report_v0.query import (
    generate_with_context,
    knn,
)


class SimpleChunkConfig(Record):
    max_chunk_size: int
    stride: int


@dataclass
class MockChunkConfig:
    chunks: list[str]


ChunkConfig = SimpleChunkConfig | MockChunkConfig

from typing import List, NoReturn, Tuple


def assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)


def chunk_corpus(corpus: str, chunk_config: ChunkConfig) -> List[str]:
    match chunk_config:
        case SimpleChunkConfig(max_chunk_size=m, stride=s):
            return simple_chunk(corpus, m, s)
        case MockChunkConfig(chunks):
            return chunks


def index_corpus(corpus: str, chunk_config: ChunkConfig) -> Tuple[List[str], types.NPA]:
    chunked = chunk_corpus(corpus, chunk_config)
    # print(f"chunked={json.dumps(list(enumerate(chunked)), indent=2)}")
    embs = np.stack([get_embedding(chunk) for chunk in chunked])

    return chunked, embs


def retrieve(query: str, chunked: List[str], embs: types.NPA) -> List[str]:
    k = 20
    top_k_idxs = knn(query, embs, "cos_sim", k)
    top_k = [chunked[i] for i in top_k_idxs]

    print(f"{top_k=}")

    return top_k


def main(argv: List[str]):
    corpus = get_raw_data()
    print(f"{corpus[:100]=}")

    mcs = 1000
    stride_ratio = 4

    stride = int(mcs / stride_ratio)
    assert stride > 0

    chunked, embs = index_corpus(  # type: ignore [fixme]
        corpus, SimpleChunkConfig(max_chunk_size=mcs, stride=stride)
    )

    print(f"chunked")
    for c in chunked[:5]:
        print(f"{c[:100]}")

    queries: List[str] = list(argv[1].split(","))  # type: ignore [fixme]
    for query in queries:  # type: ignore [fixme]
        print(f"\n\n{query=}")
        top_k: List[str] = retrieve(query, chunked, embs)

        print("top k")
        for i, tk in enumerate(top_k):
            print(f"\n{i=}")
            print(f"{tk[:100]}")

        # top_k = ["emergency broadcast system"]
        resp = generate_with_context(query, top_k)

        print(f"{resp=}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
