import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import openai
from result import Err, Ok, Result
from semantic_retrieval.common.core import (
    LOGGER_FMT,
    exp_backoff,
    flatten_list,
)
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import CallbackEvent, Record
from semantic_retrieval.document.document import Document
from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    ModelHandle,
    VectorEmbedding,
)
from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
    safe_serialize_arbitrary_for_logging,
)
from semantic_retrieval.utils.text import (
    num_tokens_from_string_for_model,
    truncate_string_to_tokens,
)
from tiktoken import encoding_for_model

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class EmbedFragmentData(Record):
    document_id: str
    fragment_id: str
    text: str


class OpenAIEmbeddingsConfig(Record):
    api_key: Optional[str] = None


class OpenAIEmbeddingsHandle(ModelHandle):
    creator: Any = openai.Embedding


DEFAULT_MODEL = "text-embedding-ada-002"

MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
}

from concurrent.futures import ThreadPoolExecutor


def _create_embeddings_with_backoff(
    input: List[str], model_handle: ModelHandle, model: str
) -> Result[Dict[str, Any], str]:
    @exp_backoff(max_retries=5, base_delay=1, logger=logger)
    def _create_embeddings() -> Dict[str, Any]:
        return model_handle.creator.create(input=input, model=model)

    return _create_embeddings()


def _make_emb_request(
    fragments: List[EmbedFragmentData],
    model: str,
    model_handle: ModelHandle,
) -> List[VectorEmbedding]:
    input = [fragment.text for fragment in fragments]

    res_embeddings = _create_embeddings_with_backoff(
        input, model_handle, model
    )
    _log = safe_serialize_arbitrary_for_logging(
        {"res_embeddings": res_embeddings}, max_elements=3
    )

    match res_embeddings:
        case Err(err):
            # TODO propagate err
            raise Exception(f"Error creating embeddings: {err}")
        case Ok(embeddings):
            logger.debug(f"Got embeddings for batch:\n{_log}")

            def _make_vector_embedding(
                frag_emb_pair: Tuple[EmbedFragmentData, Dict[str, Any]]
            ) -> VectorEmbedding:
                frag, embedding_data = frag_emb_pair
                return VectorEmbedding(
                    vector=embedding_data["embedding"],
                    text=frag.text,
                    attributes={},
                    metadata={
                        "document_id": frag.document_id,
                        "fragment_id": frag.fragment_id,
                        "model": model,
                        "text": frag.text,
                    },
                )

            pairs = list(zip(fragments, embeddings["data"]))
            out = list(map(_make_vector_embedding, pairs))
            return out


def _emb_requests_thread_pool(
    frags: List[List[EmbedFragmentData]], model: str, model_handle: ModelHandle
) -> List[Any]:
    logger.debug(f"{len(frags)=}")
    logger.info(f"Fragments total len = {sum([len(frag) for frag in frags])}")
    logger.info(f"Starting thread pool to get embeddings...")
    with ThreadPoolExecutor() as executor:
        out = list(
            executor.map(
                partial(
                    _make_emb_request, model=model, model_handle=model_handle
                ),
                frags,
            )
        )

        logger.info(f"Done with thread pool (n_batches: {len(out)})")
        return out


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
        # TODO: [P1] metadata
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

    async def embed_fragments(
        self,
        fragments: List[EmbedFragmentData],
        model_handle: Optional[ModelHandle],
        max_tokens_per_call: int,
    ) -> List[VectorEmbedding]:
        model_handle = model_handle or OpenAIEmbeddingsHandle()
        # input = [fragment.text for fragment in fragments]
        batches = [[]]
        n_tokens_this_batch = 0
        for frag in fragments:
            n_tokens_frag = num_tokens_from_string_for_model(
                frag.text, self.model
            )
            if n_tokens_frag > max_tokens_per_call:
                text_to_embed = truncate_string_to_tokens(
                    frag.text, self.model, max_tokens_per_call
                )
            else:
                text_to_embed = frag.text
            if n_tokens_frag + n_tokens_this_batch > max_tokens_per_call:
                batches.append([])
                n_tokens_this_batch = 0

            batches[-1].append(
                EmbedFragmentData(
                    document_id=frag.document_id,
                    fragment_id=frag.fragment_id,
                    text=text_to_embed,
                )
            )
            n_tokens_this_batch += n_tokens_frag

        embeddings_for_batches = _emb_requests_thread_pool(
            batches, self.model, model_handle
        )
        out = flatten_list(embeddings_for_batches)

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="openai_embeddings_fragments_embedded",
                data=dict(
                    n_batches=len(batches),
                    n_embeddings=len(out),
                ),
            )
        )

        return out

    async def transform_documents(
        self,
        documents: List[Document],
        model_handle: Optional[ModelHandle] = None,
    ) -> List[VectorEmbedding]:
        fragments = flatten_list(
            [document.fragments for document in documents]
        )
        list_embed_fragment_data = [
            EmbedFragmentData(
                document_id=fragment.document_id,
                fragment_id=fragment.fragment_id,
                text=await fragment.get_content(),
            )
            for fragment in fragments
        ]
        embeddings = await self.embed_fragments(
            list_embed_fragment_data,
            model_handle,
            self.max_encoding_length,
        )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="openai_embeddings_documents_transformed",
                data=dict(
                    n_documents=len(documents),
                    n_embeddings=len(embeddings),
                ),
            )
        )

        return embeddings
