import os
from typing import Any, Callable, Dict, List, Optional
import openai

from tiktoken import encoding_for_model
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import Record

# from openai import OpenAI, EmbeddingsResponse

from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    ModelHandle,
    VectorEmbedding,
)

from semantic_retrieval.document.document import Document


class EmbedFragmentData(Record):
    document_id: str
    fragment_id: str
    text: str


class OpenAIEmbeddingsConfig(Record):
    api_key: Optional[str] = None


class OpenAIEmbeddingsHandle(ModelHandle):
    creator: Callable[[Any], Any] = openai.Embedding


DEFAULT_MODEL = "text-embedding-ada-002"

MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings(DocumentEmbeddingsTransformer):
    model = DEFAULT_MODEL

    # TODO: Handle this for other models when they are supported
    max_encoding_length = 8191

    def __init__(self, config: OpenAIEmbeddingsConfig):
        super().__init__(MODEL_DIMENSIONS[DEFAULT_MODEL])

        api_key = config.api_key if config else os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found for OpenAIEmbeddings")

        # TODO WARNING: GLOBAL STATE MUTATION
        openai.api_key = api_key

    async def embed(
        self,
        text: str,
        model_handle: Optional[ModelHandle] = None,
        metadata: Optional[JSONObject] = None,
    ) -> VectorEmbedding:
        if not model_handle:
            model_handle = OpenAIEmbeddingsHandle()
        encoding = encoding_for_model(self.model)
        text_encoding = encoding.encode(text)

        if len(text_encoding) > self.max_encoding_length:
            # encoding.free()
            raise Exception(
                f"Text encoded length {len(text_encoding)} exceeds max input tokens {self.max_encoding_length} for model {self.model}"
            )

        # TODO wat
        # encoding.free()

        # TODO type this better
        embedding_res: Dict[Any, Any] = model_handle.creator.create(input=[text], model=self.model).to_dict_recursive()  # type: ignore
        # TODO: include usage, metadata
        return VectorEmbedding(
            vector=embedding_res["data"][0]["embedding"],
            text=text,
            # metadata={
            #     **embedding_metadata,
            #     "usage": usage,
            #     **metadata,
            #     "model": self.model,
            # },
            attributes={},
        )

    async def transform_documents(self, documents: List[Document]) -> List[VectorEmbedding]:  # type: ignore [fixme]
        # Use this to batch create embeddings with openai - https://platform.openai.com/docs/api-reference/embeddings/create
        # Just pass an array for input instead of a single string

        # Get all fragments into a list and batch create embeddings - issue is keeping the metadata in attrbutes from the original document

        # Use fragments from transformed documents to create embeddings, can use createEmbeddings as a helper here
        # Also see openAIEmbeddings.ts
        for document in documents:
            fragments = document.fragments
            for fragment in fragments:
                _content = await fragment.get_content()
                # fragment.document_id
                # fragment.fragment_id
                pass

        # Then it should do the batch logic kinda like this: https://github.com/run-llama/llama_index/blob/408923fafbcefdabfd76c8fa609b570fe80b1b2f/llama_index/embeddings/base.py#L231

        return []

    async def create_embeddings(self, fragments: List[EmbedFragmentData]) -> List[VectorEmbedding]:  # type: ignore [fixme]
        # TODO
        pass
