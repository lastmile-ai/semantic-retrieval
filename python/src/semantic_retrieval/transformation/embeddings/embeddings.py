from typing import List

from semantic_retrieval.common.base import Attributable

from semantic_retrieval.transformation.transformer import Transformer

from semantic_retrieval.common.json_types import JSONObject

from semantic_retrieval.document.document import Document, DocumentFragment


class VectorEmbedding(Attributable):
    def __init__(self):
        self.vector = []  # The vector representation of the embedding text.
        self.text = ""  # The text embedded via the vector.
        self.extras = {
            "dimensions": 0,
            "min": 0,
            "max": 0,
        }  # Number of dimensions in the vector, and min/max values for each dimension.
        self.metadata = {"documentId": "", "fragmentId": "", "retrievalScore": 0.0}  # Metadata


class EmbeddingsTransformer(Transformer):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    async def embed(self, text: str, metadata: JSONObject) -> VectorEmbedding:  # type: ignore [fixme]
        pass


class DocumentEmbeddingsTransformer(EmbeddingsTransformer):
    def __init__(self, dimensions: int):
        super().__init__(dimensions)

    async def embed(self, text: str, metadata: JSONObject) -> VectorEmbedding:  # type: ignore [fixme]
        pass

    async def embedFragment(self, fragment: DocumentFragment) -> VectorEmbedding:
        text = await fragment.get_content()
        metadata = {
            **(fragment.metadata or {}),
            "documentId": fragment.document_id,
            "fragmentId": fragment.fragment_id,
        }
        return await self.embed(text, metadata)

    async def embedDocument(self, document: Document) -> List[VectorEmbedding]:
        embeddings = []
        for fragment in document.fragments:
            embeddings.append(await self.embedFragment(fragment))
        return embeddings

    async def transformDocuments(self, documents: List[Document]) -> List[VectorEmbedding]:
        embeddings = []
        for document in documents:
            embeddings.extend(await self.embedDocument(document))
        return embeddings
