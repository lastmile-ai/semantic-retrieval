from typing import Generic, List

from attr import dataclass
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.common.types import P, Q, R
from semantic_retrieval.document.document import Document
from semantic_retrieval.generator.completion_generator import (
    LLMCompletionGeneratorParams,
    LLMCompletionGenerator,
)
from semantic_retrieval.prompts.prompt_template import PromptTemplate
from semantic_retrieval.retrieval.document_retriever import DocumentRetriever


@dataclass
class RAGCompletionGeneratorParams(Generic[P, Q], LLMCompletionGeneratorParams[P]):
    retriever: DocumentRetriever[List[Document], Q]
    accessPassport: AccessPassport
    ragPromptTemplate: PromptTemplate


DEFAULT_RAG_TEMPLATE = "Answer the question based on the context below.\n\nContext:\n\n{{context}}\n\nQuestion: {{prompt}}\n\nAnswer:"


class T_(Generic[P, Q], RAGCompletionGeneratorParams[P, Q]):
    def __init__(self, a: P, b: Q):
        self.a = a
        self.b = b


class RAGCompletionGenerator(
    Generic[P, Q, R, T_[P, Q]],  # type: ignore [fixme][This might be unfixable. Limitation of py generic types]
    LLMCompletionGenerator[P, R],
):
    async def get_retrieval_query(self, params: T_[P, Q]) -> Q:  # type: ignore [fixme]
        # TODO: Implement this method to construct the query for the underlying retriever
        pass

    async def run(  # type: ignore [fixme][This might be unfixable. Limitation of py generic types]
        self, params: T_[P, Q]
    ) -> R:  # type: ignore [fixme]
        # TODO impl
        pass


# Val = TypeVar("Val")

# class MyGeneric(Generic[Val]):
#     def __init__(self, a: Val): ...

# T = TypeVar("T")

# SingleG = MyGeneric[T]
