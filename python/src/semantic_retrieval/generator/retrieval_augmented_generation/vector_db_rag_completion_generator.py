from typing import Generic
from semantic_retrieval.common.types import P, R
from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDBBaseQuery, VectorDBQuery, VectorDBTextQuery
from semantic_retrieval.generator.retrieval_augmented_generation.rag_completion_generator import (
    RAGCompletionGenerator,
    RAGCompletionGeneratorParams,
)
from semantic_retrieval.retrieval.document_retriever import Q
from semantic_retrieval.retrieval.vector_dbs.vector_db_document_retriever import VectorDBDocumentRetriever


class VectorDBRAGCompletionGeneratorParams(
    Generic[P], RAGCompletionGeneratorParams[P, VectorDBQuery]
):
    def __init__(self, retriever: VectorDBDocumentRetriever, retrieval_query: VectorDBBaseQuery):
        self.retriever = retriever
        self.retriever_query = retrieval_query

class VectorDBRAGCompletionGenerator(
    Generic[P, R],
    RAGCompletionGenerator[
        P,
        VectorDBTextQuery,
        R,
        VectorDBRAGCompletionGeneratorParams[P],  # type: ignore [fixme][This might be unfixable. Limitation of py generic types]
    ],
):
    async def get_retrieval_query(  # type: ignore [fixme][This might be unfixable. Limitation of py generic types]
        self, params: VectorDBRAGCompletionGeneratorParams[P]
    ) -> Q:  # type: ignore [fixme]
        # TODO [P0] impl
        pass
