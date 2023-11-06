from typing import List
from semantic_retrieval.common.types import CallbackEvent

from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
    TextChunkTransformer,
    TextChunkTransformerParams,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class SeparatorTextChunkConfig(TextChunkConfig):
    separator: str = " "
    strip_new_lines: bool = True


class SeparatorTextChunkerParams(TextChunkTransformerParams):
    def __init__(self, separator_text_chunk_config: SeparatorTextChunkConfig) -> None:
        super().__init__()
        self.separator_text_chunk_config = separator_text_chunk_config


class SeparatorTextChunker(TextChunkTransformer, Traceable):
    separator: str = " "  # e.g. words
    strip_new_lines: bool = True

    def __init__(
        self,
        separator_text_chunk_config: SeparatorTextChunkConfig,
        params: TextChunkTransformerParams,
        callback_manager: CallbackManager,
    ):
        super().__init__(params, callback_manager=callback_manager)
        self.separator = separator_text_chunk_config.separator
        self.strip_new_lines = separator_text_chunk_config.strip_new_lines
        pass

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
                run_id=None,
            )
        )

        return merged_chunks
