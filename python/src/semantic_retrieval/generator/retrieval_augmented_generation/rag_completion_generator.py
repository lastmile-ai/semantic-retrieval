from abc import abstractmethod
from typing import Generic, List

from attr import dataclass
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.common.types import P, Q, R
from semantic_retrieval.document.document import Document
from semantic_retrieval.generator.completion_generator import (
    LLMCompletionGenerator,
    LLMCompletionGeneratorParams,
)
from semantic_retrieval.prompts.prompt_template import PromptTemplate
from semantic_retrieval.retrieval.document_retriever import DocumentRetriever


@dataclass
class RAGCompletionGeneratorParams(
    Generic[P, Q], LLMCompletionGeneratorParams[P]
):
    retriever: DocumentRetriever[List[Document], Q]
    accessPassport: AccessPassport
    ragPromptTemplate: PromptTemplate


DEFAULT_RAG_TEMPLATE = "Answer the question based on the context below.\n\nContext:\n\n{{context}}\n\nQuestion: {{prompt}}\n\nAnswer:"


class T_(Generic[P, Q], RAGCompletionGeneratorParams[P, Q]):
    def __init__(self, a: P, b: Q):
        self.a = a
        self.b = b


class CompletionModelResponse:
    pass


class RetrieverQuery(Generic[R]):
    pass


class RAGCompletionGenerator(
    Generic[R, P],
    LLMCompletionGenerator[P, R],
):
    @abstractmethod
    async def get_retrieval_query(self, params: P) -> RetrieverQuery[R]:
        pass

    async def run(self, params: P) -> CompletionModelResponse:  # type: ignore
        raise NotImplementedError()
