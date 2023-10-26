from dataclasses import dataclass
from typing import List, Optional, Any, Dict

# from openai import OpenAI, EmbeddingsResponse

from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    VectorEmbedding,
)

from semantic_retrieval.document.document import Document


class OpenAIClientOptions:
    # TODO: import openai
    pass


class OpenAI:
    # TODO: import
    def __init__(self, **kwargs) -> None:
        pass


def getEnvVar(key):
    return "THE_VAR_VALUE"


@dataclass
class OpenAIEmbeddingsConfig(OpenAIClientOptions):
    apiKey: Optional[str]


class OpenAIEmbeddings(DocumentEmbeddingsTransformer):
    DEFAULT_MODEL = "text-embedding-ada-002"
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: Optional[OpenAIEmbeddingsConfig] = None):
        super().__init__(self.MODEL_DIMENSIONS[self.DEFAULT_MODEL])

        apiKey = config.apiKey if config and config.apiKey else getEnvVar("OPENAI_API_KEY")
        if not apiKey:
            raise Exception("No OpenAI API key found for OpenAIEmbeddings")

        self.client = OpenAI(apiKey=apiKey, **(config or {}))

    def embed(self, text: str, metadata: Optional[Dict] = None) -> VectorEmbedding:
        encoding = encoding_for_model(self.model)
        text_encoding = encoding.encode(text)

        if len(text_encoding) > self.maxEncodingLength:
            encoding.free()
            raise Exception(
                f"Text encoded length {len(text_encoding)} exceeds max input tokens ({self.maxEncodingLength}) for model {self.model}"
            )

        encoding.free()

        embedding_res = self.client.embeddings.create(
            input=text,
            model=self.model,
        )

        data = embedding_res["data"]
        metadata_usage = embedding_res["usage"]
        embedding_metadata = {
            key: embedding_res[key] for key in embedding_res if key not in ("data", "usage")
        }
        usage = metadata_usage if len(data) == 1 else None

        return VectorEmbedding(
            vector=data[0]["embedding"],
            text=text,
            metadata={
                **embedding_metadata,
                "usage": usage,
                **metadata,
                "model": self.model,
            },
            attributes={},
        )

    async def transform_documents(self, documents: List[Document]) -> List[VectorEmbedding]:
        embeddings = []

        request_batch = []
        MAX_REQUEST_BATCH_SIZE = 5

        current_text_batch = []
        current_text_batch_size = 0
        document_idx = 0

        encoding = encoding_for_model(self.model)

        while document_idx < len(documents):
            current_document = documents[document_idx]

            current_document_fragments = [
                {
                    "documentId": fragment.documentId,
                    "fragmentId": fragment.fragmentId,
                    "text": fragment.get_content(),
                }
                for fragment in current_document.fragments
            ]

            current_document_fragment_idx = 0

            while current_document_fragment_idx < len(current_document_fragments):
                current_fragment_data = current_document_fragments[current_document_fragment_idx]
                current_fragment_encoding = encoding.encode(current_fragment_data["text"])

                if len(current_fragment_encoding) > self.maxEncodingLength:
                    encoding.free()
                    raise Exception(
                        f"Fragment {current_fragment_data['fragmentId']} encoded length {len(current_fragment_encoding)} exceeds max input tokens ({self.maxEncodingLength}) for model {self.model}"
                    )

                if (
                    current_text_batch_size + len(current_fragment_encoding)
                    > self.maxEncodingLength
                ):
                    request_batch.append(current_text_batch)

                    if len(request_batch) == MAX_REQUEST_BATCH_SIZE:
                        embedding_promises = [
                            self.create_embeddings(batch) for batch in request_batch
                        ]
                        embeddings.extend(
                            [
                                emb
                                for sublist in await Promise.all(embedding_promises)
                                for emb in sublist
                            ]
                        )
                        request_batch = []

                    current_text_batch = []
                    current_text_batch_size = 0

                current_text_batch.append(current_fragment_data)
                current_text_batch_size += len(current_fragment_encoding)
                current_document_fragment_idx += 1

            document_idx += 1

        if len(current_text_batch) > 0:
            request_batch.append(current_text_batch)

        if len(request_batch) > 0:
            embedding_promises = [self.create_embeddings(batch) for batch in request_batch]
            embeddings.extend(
                [emb for sublist in await Promise.all(embedding_promises) for emb in sublist]
            )

        encoding.free()
        return embeddings

    async def create_embeddings(self, fragments: List[Dict]) -> List[VectorEmbedding]:
        input_text = [fragment["text"] for fragment in fragments]

        embedding_res = request_with_throttle_backoff(
            lambda: self.client.embeddings.create(input=input_text, model=self.model)
        )

        data = embedding_res["data"]
        metadata_usage = embedding_res["usage"]
        metadata = {
            key: embedding_res[key] for key in embedding_res if key not in ("data", "usage")
        }
        usage = metadata_usage if len(fragments) == 1 else None

        return [
            VectorEmbedding(
                vector=embedding["embedding"],
                text=fragments[idx]["text"],
                metadata={
                    **metadata,
                    "fragmentId": fragments[idx]["fragmentId"],
                    "documentId": fragments[idx]["documentId"],
                    "model": self.model,
                    "usage": usage,
                },
                attributes={},
            )
            for idx, embedding in enumerate(data)
        ]
