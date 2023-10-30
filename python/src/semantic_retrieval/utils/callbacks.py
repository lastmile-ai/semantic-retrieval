# from dataclasses import dataclass
# from typing import Any, Awaitable, Callable, List, Sequence, Union

# from semantic_retrieval.access_control.access_identity import AccessIdentity
# from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDBBaseQuery
# from semantic_retrieval.document.document import Document, DocumentFragment, IngestedDocument
# from semantic_retrieval.generator.completion_generator import LLMCompletionGeneratorParams
# from semantic_retrieval.generator.completion_models.completion_model import CompletionModelParams
# from semantic_retrieval.transformation.embeddings.embeddings import VectorEmbedding


# # Define assertUnreachable function
# def assertUnreachable(x: Any) -> None:
#     pass


# # Define the callback event types
# from dataclasses import dataclass
# from typing import Optional, Any, List


# @dataclass
# class LoadDocumentsSuccessEvent:
#     name: str = "onLoadDocumentsSuccess"
#     rawDocuments: Optional[List[Any]] = None


# @dataclass
# class LoadDocumentsErrorEvent:
#     name: str = "onLoadDocumentsError"
#     error: Optional[BaseException] = None


# @dataclass
# class DataSourceTestConnectionSuccessEvent:
#     name: str = "onDataSourceTestConnectionSuccess"
#     code: int = 0


# @dataclass
# class DataSourceTestConnectionErrorEvent:
#     name: str = "onDataSourceTestConnectionError"
#     code: int = 0
#     error: Any = None


# @dataclass
# class ParseNextErrorEvent:
#     name: str = "onParseNextError"
#     error: Any = None


# @dataclass
# class ParseErrorEvent:
#     name: str = "onParseError"
#     error: Any = None


# @dataclass
# class ParseSuccessEvent:
#     ingestedDocument: IngestedDocument
#     name: str = "onParseSuccess"


# @dataclass
# class TransformDocumentsEvent:
#     name: str = "onTransformDocuments"
#     transformedDocuments: Optional[List[Document]] = None
#     originalDocuments: Optional[List[Document]] = None


# @dataclass
# class TransformDocumentEvent:
#     originalDocument: Document
#     transformedDocument: Document
#     name: str = "onTransformDocument"


# @dataclass
# class ChunkTextEvent:
#     name: str = "onChunkText"
#     chunks: Optional[List[str]] = None


# @dataclass
# class RegisterAccessIdentityEvent:
#     identity: AccessIdentity
#     name: str = "onRegisterAccessIdentity"


# @dataclass
# class GetAccessIdentityEvent:
#     identity: AccessIdentity
#     name: str = "onGetAccessIdentity"
#     resource: str = ""


# @dataclass
# class AddDocumentsToVectorDBEvent:
#     name: str = "onAddDocumentsToVectorDB"
#     documents: Optional[List[Document]] = None


# @dataclass
# class QueryVectorDBEvent:
#     query: VectorDBBaseQuery
#     vectorEmbeddings: Optional[List[VectorEmbedding]]
#     name: str = "onQueryVectorDB"


# @dataclass
# class RetrieverFilterAccessibleFragmentsEvent:
#     name: str = "onRetrieverFilterAccessibleFragments"
#     fragments: Optional[List[DocumentFragment]] = None


# @dataclass
# class RetrieverGetDocumentsForFragmentsEvent:
#     name: str = "onRetrieverGetDocumentsForFragments"
#     documents: Optional[List[Document]] = None


# @dataclass
# class RetrieverProcessDocumentsEvent:
#     documents: List[Document]
#     name: str = "onRetrieverProcessDocuments"


# @dataclass
# class RetrieveDataEvent:
#     name: str = "onRetrieveData"
#     data: Any = None


# @dataclass
# class GetFragmentsEvent:
#     name: str = "onGetFragments"
#     fragments: Optional[List[DocumentFragment]] = None


# @dataclass
# class RunCompletionEvent:
#     params: CompletionModelParams[Any]
#     response: Any = None
#     name: str = "onRunCompletion"


# @dataclass
# class RunCompletionGenerationEvent:
#     params: LLMCompletionGeneratorParams[Any]
#     response: Any
#     name: str = "onRunCompletionGeneration"


# @dataclass
# class GetRAGCompletionRetrievalQueryEvent:
#     name: str = "onGetRAGCompletionRetrievalQuery"
#     params: Any = None
#     query: Any = None


# # Define CallbackEvent union type
# CallbackEvent = Union[
#     LoadDocumentsSuccessEvent,
#     LoadDocumentsErrorEvent,
#     DataSourceTestConnectionSuccessEvent,
#     DataSourceTestConnectionErrorEvent,
#     ParseNextErrorEvent,
#     ParseErrorEvent,
#     ParseSuccessEvent,
#     TransformDocumentsEvent,
#     TransformDocumentEvent,
#     ChunkTextEvent,
#     RegisterAccessIdentityEvent,
#     GetAccessIdentityEvent,
#     AddDocumentsToVectorDBEvent,
#     QueryVectorDBEvent,
#     RetrieverFilterAccessibleFragmentsEvent,
#     RetrieverGetDocumentsForFragmentsEvent,
#     RetrieverProcessDocumentsEvent,
#     RetrieveDataEvent,
#     GetFragmentsEvent,
#     RunCompletionEvent,
#     RunCompletionGenerationEvent,
#     GetRAGCompletionRetrievalQueryEvent,
# ]

# Callback = Callable[[CallbackEvent, str], Awaitable[None]]


# @dataclass
# class CallbackManager:
#     runId: str
#     callbacks: dict[str, Sequence[Callback]]

#     async def run_callbacks(self, event: CallbackEvent) -> None:
#         # TODO
#         pass


# # Define Traceable interface
# class Traceable:
#     callback_manager: Optional[CallbackManager] = None


# @dataclass
# class CallbackMapping:
#     onLoadDocumentsSuccess: Optional[List[Any]] = None
#     onLoadDocumentsError: Optional[List[Any]] = None
#     onDataSourceTestConnectionSuccess: Optional[List[Any]] = None
#     onDataSourceTestConnectionError: Optional[List[Any]] = None
#     onParseNextError: Optional[List[Any]] = None
#     onParseError: Optional[List[Any]] = None
#     onParseSuccess: Optional[List[Any]] = None
#     onTransformDocuments: Optional[List[Any]] = None
#     onTransformDocument: Optional[List[Any]] = None
#     onChunkText: Optional[List[Any]] = None
#     onRegisterAccessIdentity: Optional[List[Any]] = None
#     onGetAccessIdentity: Optional[List[Any]] = None
#     onAddDocumentToVectorDB: Optional[List[Any]] = None
#     onQueryVectorDB: Optional[List[Any]] = None
#     onRetrieverFilterAccessibleFragments: Optional[List[Any]] = None
#     onRetrieverGetDocumentsForFragments: Optional[List[Any]] = None
#     onRetrieverProcessDocuments: Optional[List[Any]] = None
#     onRetrieveData: Optional[List[Any]] = None
#     onGetFragments: Optional[List[Any]] = None
#     onRunCompletion: Optional[List[Any]] = None
#     onRunCompletionGeneration: Optional[List[Any]] = None
#     onGetRAGCompletionRetrievalQuery: Optional[List[Any]] = None


# # Define DEFAULT_CALLBACKS
# DEFAULT_CALLBACKS = {"onLoadDocumentsSuccess": []}


from typing import Any, Awaitable, Callable, Final, Optional, Sequence

from semantic_retrieval.common.types import Record


class Traceable:
    """
    Interface for classes that support callbacks.

    TODO: figure out a way to type-enforce this
    By extending Traceable, a class affirms that it
    * accepts and stores a CallbackManager on init,
    * calls `run_callbacks()` on the CallbackManager
      with the appropriate event arguments at the appropriate
      points in its implementation.
    """

    pass


class CallbackEvent(Record):
    # Anything available at the time the event happens.
    # It is passed to the callback.
    data: Any
    # Globally unique identifier for the (e2e) run.
    # Callbacks should include this in any logs written,
    # as it is necessary to stitch together separate
    # records for analysis.
    run_id: str


class CallbackResult(Record):
    result: Any


# Callbacks will run on every event.
# They may have I/O side effects (e.g. logging) and/or return a CallbackResult.
# Any CallbackResults returned will be stored in the CallbackManager.
# The user can then access these results.
Callback = Callable[[CallbackEvent], Awaitable[Optional[CallbackResult]]]


class CallbackManager:
    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self.callbacks: Final[Sequence[Callback]] = callbacks
        self.results = []

    # TODO: statically type each event?
    # TODO: [optimization] index callbacks by event type?
    async def run_callbacks(self, event: CallbackEvent) -> Optional[CallbackResult]:
        # TODO
        # TODO: [optimization] do this storage more efficiently
        for callback in self.callbacks:
            result = await callback(event)
            if result is not None:
                self.results.append(result)
