from typing import List, Optional

from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
    TextChunkTransformer,
    TextChunkTransformerParams,
)

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class SeparatorTextChunkConfig(TextChunkConfig):
    separator: str


class SeparatorTextChunkerParams(TextChunkTransformerParams, SeparatorTextChunkConfig):
    pass


class SeparatorTextChunker(TextChunkTransformer):
    separator: str = " "  # e.g. words

    def __init__(self, params: Optional[SeparatorTextChunkerParams] = None):
        super().__init__(params)
        self.separator = (
            params.separator if params and hasattr(params, "separator") else self.separator
        )

    async def chunk_text(self, text: str) -> List[str]:
        sub_chunks = self.sub_chunk_on_separator(text, self.separator)
        merged_chunks = await self.merge_sub_chunks(sub_chunks, self.separator)

        event = ChunkTextEvent(name="onChunkText", chunks=merged_chunks)
        if self.callback_manager:
            self.callback_manager.run_callbacks(event)

        return merged_chunks
