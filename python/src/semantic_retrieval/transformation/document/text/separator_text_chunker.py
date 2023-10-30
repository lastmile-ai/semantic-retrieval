from typing import List

from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
    TextChunkTransformer,
    TextChunkTransformerParams,
)


class SeparatorTextChunkConfig(TextChunkConfig):
    separator: str = " "


class SeparatorTextChunkerParams(TextChunkTransformerParams):
    separator_text_chunk_config: SeparatorTextChunkConfig


class SeparatorTextChunker(TextChunkTransformer):
    separator: str = " "  # e.g. words

    def __init__(self, params: SeparatorTextChunkerParams):
        super().__init__(params)
        # TODO
        pass

    async def chunk_text(self, text: str) -> List[str]:
        # TODO impl
        return []
