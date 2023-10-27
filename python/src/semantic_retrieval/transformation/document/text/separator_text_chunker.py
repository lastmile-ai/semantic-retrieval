from dataclasses import dataclass, field
from typing import List, Optional

from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
    TextChunkTransformer,
    TextChunkTransformerParams,
)

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


@dataclass
class SeparatorTextChunkConfig(TextChunkConfig):
    separator: str = " "


@dataclass
class SeparatorTextChunkerParams:
    text_chunk_transformer_params: TextChunkTransformerParams = field(
        default_factory=TextChunkTransformerParams
    )
    separator_text_chunk_config: SeparatorTextChunkConfig = field(
        default_factory=SeparatorTextChunkConfig
    )


class SeparatorTextChunker(TextChunkTransformer):
    separator: str = " "  # e.g. words

    def __init__(self, params: Optional[SeparatorTextChunkerParams] = None):
        super().__init__(params)
        # TODO
        pass

    async def chunk_text(self, text: str) -> List[str]:
        # TODO
        pass
