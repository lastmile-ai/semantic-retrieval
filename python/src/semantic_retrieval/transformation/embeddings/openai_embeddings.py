import os
from typing import Any, Dict, List, Optional
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
        model_handle: ModelHandle,
        text: str,
        metadata: Optional[JSONObject] = None,
    ) -> VectorEmbedding:
        encoding = encoding_for_model(self.model)
        text_encoding = encoding.encode(text)

        if len(text_encoding) > self.max_encoding_length:
            # encoding.free()
            raise Exception(
                f"Text encoded length {len(text_encoding)} exceeds max input tokens {self.max_encoding_length} for model {self.model}"
            )

        # TODO wat
        # encoding.free()

        embedding_res: Dict[Any, Any] = model_handle.create(input=[text], model=self.model).to_dict_recursive()  # type: ignore [fixme]

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
        # TODO
        pass

    async def create_embeddings(self, fragments: List[EmbedFragmentData]) -> List[VectorEmbedding]:  # type: ignore [fixme]
        # TODO
        pass
