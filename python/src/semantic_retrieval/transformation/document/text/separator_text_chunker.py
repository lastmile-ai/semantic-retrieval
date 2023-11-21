from dataclasses import dataclass
from typing import List, Optional

from semantic_retrieval.common.types import CallbackEvent
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkTransformer,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


@dataclass
class SeparatorTextChunkerParams:
    document_metadata_db: Optional[DocumentMetadataDB]
    separator: str = " "
    strip_new_lines: bool = True
    chunk_size_limit: int = 500
    chunk_overlap: int = 100


class SeparatorTextChunker(TextChunkTransformer, Traceable):
    separator: str = " "  # e.g. words
    strip_new_lines: bool = True

    def __init__(
        self,
        params: SeparatorTextChunkerParams,
        callback_manager: CallbackManager,
    ):
        super().__init__(callback_manager=callback_manager)
        self.callback_manager = callback_manager
        self.document_metadata_db = params.document_metadata_db
        self.separator = params.separator
        self.strip_new_lines = params.strip_new_lines
        self.chunk_size_limit = params.chunk_size_limit
        self.chunk_overlap = params.chunk_overlap

    async def chunk_text(self, text: str) -> List[str]:
        text_to_chunk = text
        if self.strip_new_lines:
            text_to_chunk = text_to_chunk.replace("\n", " ")

        sub_chunks = self.sub_chunk_on_separator(text_to_chunk, self.separator)
        merged_chunks = await self.merge_sub_chunks(sub_chunks, self.separator)

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="chunk_text",
                data=dict(
                    text=text,
                    separator=self.separator,
                    strip_new_lines=self.strip_new_lines,
                    sub_chunks=sub_chunks,
                    merged_chunks=merged_chunks,
                ),
            )
        )

        return merged_chunks
