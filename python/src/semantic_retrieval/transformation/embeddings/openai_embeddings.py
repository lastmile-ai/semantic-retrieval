import os
from typing import Any, Callable, Dict, List, Optional
import openai

from tiktoken import encoding_for_model
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import CallbackEvent, Record

# from openai import OpenAI, EmbeddingsResponse

from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    ModelHandle,
    VectorEmbedding,
)

from semantic_retrieval.document.document import Document
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


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


class OpenAIEmbeddings(DocumentEmbeddingsTransformer, Traceable):
    model = DEFAULT_MODEL

    # TODO [P1]: Handle this for other models when they are supported
    max_encoding_length = 8191

    def __init__(
        self, config: OpenAIEmbeddingsConfig, callback_manager: CallbackManager
    ):
        super().__init__(
            MODEL_DIMENSIONS[DEFAULT_MODEL], callback_manager=callback_manager
        )

        self.callback_manager = callback_manager

        api_key = config.api_key if config else os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found for OpenAIEmbeddings")

        # TODO [P1] WARNING: GLOBAL STATE MUTATION
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

        # TODO [P1] wat
        # encoding.free()

        # TODO [P1] type this better
        embedding_res: Dict[Any, Any] = model_handle.creator.create(input=[text], model=self.model).to_dict_recursive()  # type: ignore
        # TODO: [P1] include usage
        # TODO: [P0.5] metadata
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

    async def transform_documents(
        self, documents: List[Document], model_handle: Optional[ModelHandle] = None
    ) -> List[VectorEmbedding]:
        # TODO [P0]: Update this to batch embeddings instead of creating one at a time
        # Use this to batch create embeddings with openai - https://platform.openai.com/docs/api-reference/embeddings/create
        # See: https://github.com/run-llama/llama_index/blob/408923fafbcefdabfd76c8fa609b570fe80b1b2f/llama_index/embeddings/base.py#L231
        # Also see openAIEmbeddings.ts
        embeddings = []
        for document in documents:
            fragments = document.fragments
            for fragment in fragments:
                # Instead of batching, just create embeddings for each fragment right now, batching can be done as optimization
                # Need to essentially count tokens & add to array
                content = await fragment.get_content()
                vec_embeddings = await self.create_embeddings(
                    [
                        EmbedFragmentData(
                            document_id=fragment.document_id,
                            fragment_id=fragment.fragment_id,
                            text=content,
                        )
                    ]
                )

                embeddings.extend(vec_embeddings)

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="openai_embeddings_documents_transformed",
                data=dict(
                    documents=documents,
                    embeddings=embeddings,
                ),
                run_id=None,
            )
        )

        return embeddings

    async def create_embeddings(self, fragments: List[EmbedFragmentData]) -> List[VectorEmbedding]:  # type: ignore [fixme]
        model_handle = OpenAIEmbeddingsHandle()

        input = [fragment.text for fragment in fragments]

        # TODO [P0]: This is very slow... need to batch this & make this async (acreate)
        embeddings = model_handle.creator.create(input=input, model=self.model)  # type: ignore

        vector_embeddings: List[VectorEmbedding] = []
        for idx, embedding in enumerate(embeddings["data"]):
            vector_embeddings.append(
                VectorEmbedding(
                    vector=embedding["embedding"],
                    text=fragments[idx].text,
                    attributes={},
                    metadata={
                        "document_id": fragments[idx].document_id,
                        "fragment_id": fragments[idx].fragment_id,
                        "model": self.model,
                    },
                )
            )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="openai_embeddings_created",
                data=dict(
                    fragments=fragments,
                    embeddings=vector_embeddings,
                ),
                run_id=None,
            )
        )

        return vector_embeddings
