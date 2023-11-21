from hashlib import md5
from typing import List, Optional
from uuid import uuid4

from semantic_retrieval.common.types import CallbackEvent, Record
from semantic_retrieval.document.document import (
    Document,
    DocumentFragmentType,
    TransformedDocument,
)
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.transformation.document.document_transformer import (
    BaseDocumentTransformer,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class TextChunkConfig(Record):
    chunk_size_limit: int
    chunk_overlap: int


class TextChunkTransformerParams:
    separator: str = " "
    strip_new_lines: bool = True
    metadata_db: Optional[DocumentMetadataDB]


async def _len(x: str) -> int:
    return len(x)


class TextChunkTransformer(BaseDocumentTransformer, Traceable):
    def __init__(self, callback_manager: CallbackManager):
        self.size_fn = _len
        self.chunk_size_limit = 500
        self.chunk_overlap = 100
        self.callback_manager = callback_manager

    async def chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError(
            "This method must be implemented in a derived class"
        )

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

            def id_(chunk: str) -> str:
                return chunk

            # TODO [P1]: One other issue with chunks is some csvs are getting 0 chunks returned even though they have sub_chunks
            # Definitely some issue with merge_sub_chunks, but not going to debug this right now
            if not isinstance(original_fragment, str):
                raise ValueError(
                    f"Expected original_fragment to be a string, but got {type(original_fragment)}"
                )
            for chunk in await self.chunk_text(original_fragment):
                current_fragment = {
                    "fragment_id": str(uuid4()),
                    "fragment_type": DocumentFragmentType.TEXT,
                    "document_id": document_id,
                    "metadata": original_fragment_data["metadata"],
                    "attributes": {},
                    "hash": md5(chunk.encode()).hexdigest(),
                    "content": chunk,
                    "getContent": id_,
                    "serialize": id_,
                }

                # TODO [P1]: Unsure if this is working correctly
                if fragment_count > 0:
                    transformed_fragments[fragment_count - 1][
                        "nextFragment"
                    ] = current_fragment

                fragment_count += 1
                transformed_fragments.append(current_fragment)

        transformed_document = TransformedDocument(
            document=document,
            document_id=document_id,
            fragments=transformed_fragments,
            collection_id=document.collection_id,
            metadata={},
            attributes={},
        )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="text_chunk_transform_document",
                data=dict(
                    original_document=document,
                    transformed_document=transformed_document,
                    original_fragments_data=original_fragments_data,
                ),
            )
        )

        return transformed_document

    def join_sub_chunks(
        self, sub_chunks: List[str], separator: str
    ) -> Optional[str]:
        chunk = separator.join(sub_chunks).strip()
        return chunk if chunk != "" else None

    async def merge_sub_chunks(
        self, sub_chunks: List[str], separator: str
    ) -> List[str]:
        chunks = []
        prev_sub_chunks = []
        current_sub_chunks = []
        current_chunk_size = 0
        chunk_separator_size = await self.size_fn(separator)

        for sub_chunk in sub_chunks:
            sub_chunk_size = await self.size_fn(sub_chunk)

            if sub_chunk_size > self.chunk_size_limit:
                print(
                    f"SubChunk size {sub_chunk_size} exceeds chunkSizeLimit of {self.chunk_size_limit}"
                )

            if (
                current_chunk_size + chunk_separator_size + sub_chunk_size
                > self.chunk_size_limit
            ):
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
                            num_total_prev_sub_chunks
                            - num_prev_sub_chunks_overlap
                            - 1
                        ]
                    )

                    if (
                        prev_sub_chunks_overlap_size
                        + chunk_separator_size
                        + next_prev_sub_chunk_size
                        > self.chunk_overlap
                    ) or (
                        prev_sub_chunks_overlap_size
                        + chunk_separator_size
                        + next_prev_sub_chunk_size
                        + sub_chunk_size
                        > self.chunk_size_limit
                    ):
                        break

                    prev_sub_chunks_overlap_size += next_prev_sub_chunk_size

                    if num_prev_sub_chunks_overlap > 0:
                        prev_sub_chunks_overlap_size += chunk_separator_size

                    num_prev_sub_chunks_overlap += 1

                while num_prev_sub_chunks_overlap > 0:
                    current_sub_chunks.append(
                        prev_sub_chunks[
                            num_total_prev_sub_chunks
                            - num_prev_sub_chunks_overlap
                        ]
                    )
                    num_prev_sub_chunks_overlap -= 1

                current_chunk_size = prev_sub_chunks_overlap_size

            current_sub_chunks.append(sub_chunk)

            if len(current_sub_chunks) > 1:
                current_chunk_size += chunk_separator_size

            current_chunk_size += sub_chunk_size

        if len(current_sub_chunks) > 0:
            chunk = self.join_sub_chunks(current_sub_chunks, separator)

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="merge_sub_chunks",
                data=dict(
                    sub_chunks=sub_chunks,
                    separator=separator,
                    chunks=chunks,
                ),
            )
        )

        # TODO [P1] is this correct?
        return chunks
