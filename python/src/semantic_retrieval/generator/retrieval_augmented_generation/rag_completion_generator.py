from typing import Awaitable, Generic
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.common.types import Q, R, T
from semantic_retrieval.generator.completion_generator import (
    LLMCompletionGeneratorParams,
    LLMCompletionGenerator,
)
from semantic_retrieval.retrieval.retriever import BaseRetriever


class RAGCompletionGeneratorParams(LLMCompletionGeneratorParams):
    def __init__(
        self,
        retriever: BaseRetriever,
        accessPassport: AccessPassport = None,
        ragPromptTemplate=None,
    ):
        self.retriever = retriever
        self.accessPassport = accessPassport
        self.ragPromptTemplate = ragPromptTemplate


DEFAULT_RAG_TEMPLATE = "Answer the question based on the context below.\n\nContext:\n\n{{context}}\n\nQuestion: {{prompt}}\n\nAnswer:"


class RAGCompletionGenerator(LLMCompletionGenerator, Generic[T]):
    async def get_retrieval_query(self, params: T) -> Q:
        # Implement this method to construct the query for the underlying retriever
        pass

    async def run(self, params: T) -> R:
        # TODO
        pass
