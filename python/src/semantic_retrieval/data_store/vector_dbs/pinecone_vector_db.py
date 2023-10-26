from uuid import uuid4 as uuid
from typing import Any, List

from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDB, VectorDBConfig

from semantic_retrieval.document.document import Document


def getEnvVar(k):
    # TODO
    return "VALUE"


class PineconeVectorDB(VectorDB):
    def __init__(self, config: VectorDBConfig):
        # TODO
        return
        super().__init__(config["embeddings"], config["metadataDB"])

        apiKey = config.get("apiKey", getEnvVar("PINECONE_API_KEY"))
        if not apiKey:
            raise Exception("No Pinecone API key found for PineconeVectorDB")

        environment = config.get("environment", getEnvVar("PINECONE_ENVIRONMENT"))
        if not environment:
            raise Exception("No Pinecone environment found for PineconeVectorDB")

        self.client = Pinecone(apiKey=apiKey, environment=environment)
        self.index = self.client.index(config["indexName"])

        # Default namespace is an empty string
        self.namespace = self.index.namespace(config.namespace if config.namespace else "")

        # If the dimensions don't match, future operations will error out, so warn as soon as possible
        stats = self.index.describe_index_stats()
        if stats.dimension and stats.dimension != self.embeddings.dimensions:
            print(
                f"PineconeVectorDB embedding dimensions ({self.embeddings.dimensions}) do not match index dimensions ({stats.dimension})"
            )

    @classmethod
    def from_documents(cls, documents: List[Document], config: VectorDBConfig):
        instance = cls(config)
        instance.add_documents(documents)
        return instance

    def sanitize_metadata(self, unsanitized_metadata):
        string_array_metadata = {}
        mutable_metadata = dict(unsanitized_metadata)

        def set_string_array_metadata_recursive(metadata, key_path=[]):
            for key, value in metadata.items():
                updated_key_path = key_path + [key]
                if isinstance(value, list) and all(isinstance(v, str) for v in value):
                    string_array_metadata[".".join(updated_key_path)] = value
                    del metadata[key]
                elif isinstance(value, dict):
                    set_string_array_metadata_recursive(value, updated_key_path)

        set_string_array_metadata_recursive(mutable_metadata)

        metadata = {**flatten(mutable_metadata), **string_array_metadata}

        # Remove nulls since Pinecone does not support null values
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return metadata

    def add_documents(self, documents):
        # TODO
        return
        embeddings = self.embeddings.transform_documents(documents)
        pinecone_vectors = []

        for embedding in embeddings:
            metadata = {**embedding.metadata, **embedding.attributes, "text": embedding.text}
            sanitized_metadata = self.sanitize_metadata(metadata)
            pinecone_vectors.append(
                PineconeRecord(id=uuid(), values=embedding.vector, metadata=sanitized_metadata)
            )

        # Pinecone recommends a limit of 100 vectors max per upsert request
        VECTORS_PER_REQUEST = 80
        PARALLEL_REQUESTS = 5
        vector_idx = 0

        while vector_idx < len(pinecone_vectors):
            requests = []
            for _ in range(PARALLEL_REQUESTS):
                vectors = pinecone_vectors[vector_idx : vector_idx + VECTORS_PER_REQUEST]
                if vectors:
                    requests.append(
                        requestWithThrottleBackoff(lambda: self.namespace.upsert(vectors))
                    )
                    vector_idx += VECTORS_PER_REQUEST

            for request in requests:
                request()

        event = AddDocumentsToVectorDBEvent(name="onAddDocumentsToVectorDB", documents=documents)
        self.callback_manager.run_callbacks(event)

    def query(self, query):
        if isEmbeddingQuery(query):
            query_vector = query.embedding_vector.vector
        else:
            text = query.text
            query_vector = self.embeddings.embed(text).vector

        results = self.namespace.query(
            include_metadata=True,
            top_k=query.topK,
            vector=query_vector,
            filter=query.metadata_filter,
        )

        vector_embeddings = []
        for match in results.matches:
            metadata = unflatten({**match.metadata})
            attributes = metadata.get("attributes", {})
            text = metadata.pop("text", "")
            vector_embeddings.append(
                VectorEmbedding(
                    vector=match.values,
                    text=text,
                    metadata={**metadata, "retrievalScore": match.score},
                    attributes=attributes,
                )
            )

        event = QueryVectorDBEvent(
            name="onQueryVectorDB", query=query, vector_embeddings=vector_embeddings
        )
        self.callback_manager.run_callbacks(event)

        return vector_embeddings
