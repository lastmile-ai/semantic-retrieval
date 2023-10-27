from uuid import uuid4
from hashlib import md5
from typing import List, Optional

from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.transformation.document.document_transformer import (
    BaseDocumentTransformer,
)

from semantic_retrieval.document.document import Document, RawDocument


class TextChunkConfig:
    chunkSizeLimit: int
    chunkOverlap: int
    sizeFn: None


class TextChunkTransformerParams(TextChunkConfig):
    metadata_db: Optional[DocumentMetadataDB]


class TextChunkTransformer(BaseDocumentTransformer, TextChunkConfig):
    def __init__(self, params: Optional[TextChunkTransformerParams] = None):
        # TODO
        pass

    async def chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError("This method must be implemented in a derived class")

    def sub_chunk_on_separator(self, text: str, separator: str) -> List[str]:
        sub_chunks = text.split(separator)
        return [sc for sc in sub_chunks if sc != ""]

    async def transform_document(self, document: Document) -> Document:
        original_fragments_data = [
            {
                "attributes": fragment.attributes,
                "content": await fragment.get_content(),
                "fragmentType": fragment.fragment_type,
                "metadata": fragment.metadata,
            }
            for fragment in document.fragments
        ]

        transformed_fragments = []
        fragment_count = 0
        document_id = str(uuid4())

        for original_fragment_data in original_fragments_data:
            original_fragment = original_fragment_data["content"]

            for chunk in await self.chunk_text(original_fragment):
                current_fragment = {
                    "fragmentId": str(uuid4()),
                    "fragmentType": "text",
                    "documentId": document_id,
                    "metadata": original_fragment_data["metadata"],
                    "attributes": {},
                    "hash": md5(chunk.encode()).hexdigest(),
                    "getContent": lambda chunk=chunk: chunk,
                    "serialize": lambda chunk=chunk: chunk,
                }

                if fragment_count > 0:
                    transformed_fragments[fragment_count - 1]["nextFragment"] = current_fragment

                fragment_count += 1
                transformed_fragments.append(current_fragment)

        transformed_document = {
            **document,
            "documentId": document_id,
            "fragments": transformed_fragments,
        }

        event = TransformDocumentEvent(
            name="onTransformDocument",
            originalDocument=document,
            transformedDocument=transformed_document,
        )

        if self.callback_manager:
            self.callback_manager.run_callbacks(event)

        return transformed_document

    def join_sub_chunks(self, sub_chunks: List[str], separator: str) -> Optional[str]:
        chunk = separator.join(sub_chunks).strip()
        return chunk if chunk != "" else None

    async def merge_sub_chunks(self, sub_chunks: List[str], separator: str) -> List[str]:
        chunks = []
        prev_sub_chunks = []
        current_sub_chunks = []
        current_chunk_size = 0
        chunk_separator_size = await self.size_fn(separator)

        for sub_chunk in sub_chunks:
            sub_chunk_size = await self.size_fn(sub_chunk)

            if sub_chunk_size > self.chunkSizeLimit:
                print(
                    f"SubChunk size {sub_chunk_size} exceeds chunkSizeLimit of {self.chunkSizeLimit}"
                )

            if current_chunk_size + chunk_separator_size + sub_chunk_size > self.chunkSizeLimit:
                chunk = self.join_sub_chunks(current_sub_chunks, separator)
                if chunk is not None:
                    chunks.append(chunk)
                prev_sub_chunks = current_sub_chunks
                current_chunk_size = 0
                current_sub_chunks = []

            num_total_prev_sub_chunks = len(prev_sub_chunks)

            if len(current_sub_chunks) == 0 and num_total_prev_sub_chunks > 0:
                prev_sub_chunks_overlap_size = 0
                num_prev_sub_chunks_overlap = 0

                while num_prev_sub_chunks_overlap < num_total_prev_sub_chunks:
                    next_prev_sub_chunk_size = await self.size_fn(
                        prev_sub_chunks[
                            num_total_prev_sub_chunks - num_prev_sub_chunks_overlap - 1
                        ]
                    )

                    if (
                        prev_sub_chunks_overlap_size
                        + chunk_separator_size
                        + next_prev_sub_chunk_size
                        > self.chunkOverlap
                    ) or (
                        prev_sub_chunks_overlap_size
                        + chunk_separator_size
                        + next_prev_sub_chunk_size
                        + sub_chunk_size
                        > self.chunkSizeLimit
                    ):
                        break

                    prev_sub_chunks_overlap_size += next_prev_sub_chunk_size

                    if num_prev_sub_chunks_overlap > 0:
                        prev_sub_chunks_overlap_size += chunk_separator_size

                    num_prev_sub_chunks_overlap += 1

                while num_prev_sub_chunks_overlap > 0:
                    current_sub_chunks.append(
                        prev_sub_chunks[num_total_prev_sub_chunks - num_prev_sub_chunks_overlap]
                    )
                    num_prev_sub_chunks_overlap -= 1

                current_chunk_size = prev_sub_chunks_overlap_size

            current_sub_chunks.append(sub_chunk)

            if len(current_sub_chunks) > 1:
                current_chunk_size += chunk_separator_size

            current_chunk_size += sub_chunk_size

        if len(current_sub_chunks) > 0:
            chunk = self.join_sub_chunks(current_sub_chunks, separator)
