from typing import List

from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
    TextChunkTransformer,
    TextChunkTransformerParams,
)


class SeparatorTextChunkConfig(TextChunkConfig):
    separator: str = " "
    strip_new_lines: bool = True


class SeparatorTextChunkerParams(TextChunkTransformerParams):
    separator_text_chunk_config: SeparatorTextChunkConfig


class SeparatorTextChunker(TextChunkTransformer):
    separator: str = " "  # e.g. words
    strip_new_lines: bool = True

    def __init__(self, params: SeparatorTextChunkerParams):
        super().__init__(params)
        # TODO: Impl
        pass

    async def chunk_text(self, text: str) -> List[str]:
        text_to_chunk = text
        if self.strip_new_lines:
            text_to_chunk = text_to_chunk.replace("\n", " ")

        sub_chunks = self.sub_chunk_on_separator(text_to_chunk, self.separator)
        merged_chunks = await self.merge_sub_chunks(sub_chunks, self.separator)

        # TODO: Callback / events

        return merged_chunks
