from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Type, Union


# Define AccessIdentity
class AccessIdentity:
    pass


# Define assertUnreachable function
def assertUnreachable(x: Any) -> None:
    pass


# Define VectorDBQuery
class VectorDBQuery:
    pass


# Define document classes
class IngestedDocument:
    pass


class RawDocument:
    pass


class Document:
    pass


class DocumentFragment:
    pass


# Define completion model classes
class CompletionModelParams:
    pass


class LLMCompletionGeneratorParams:
    pass


# Define VectorEmbedding
class VectorEmbedding:
    pass


# Define the callback event types
from dataclasses import dataclass
from typing import List, Optional, Any

from dataclasses import dataclass
from typing import List, Optional, Any

from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class LoadDocumentsSuccessEvent:
    name: str = "onLoadDocumentsSuccess"
    rawDocuments: List = None


@dataclass
class LoadDocumentsErrorEvent:
    name: str = "onLoadDocumentsError"
    error: Optional[BaseException] = None


@dataclass
class DataSourceTestConnectionSuccessEvent:
    name: str = "onDataSourceTestConnectionSuccess"
    code: int = 0


@dataclass
class DataSourceTestConnectionErrorEvent:
    name: str = "onDataSourceTestConnectionError"
    code: int = 0
    error: Any = None


@dataclass
class ParseNextErrorEvent:
    name: str = "onParseNextError"
    error: Any = None


@dataclass
class ParseErrorEvent:
    name: str = "onParseError"
    error: Any = None


@dataclass
class ParseSuccessEvent:
    name: str = "onParseSuccess"
    ingestedDocument: IngestedDocument = IngestedDocument()


@dataclass
class TransformDocumentsEvent:
    name: str = "onTransformDocuments"
    transformedDocuments: List[Document] = None
    originalDocuments: List[Document] = None


@dataclass
class TransformDocumentEvent:
    name: str = "onTransformDocument"
    originalDocument: Document = Document()
    transformedDocument: Document = Document()


@dataclass
class ChunkTextEvent:
    name: str = "onChunkText"
    chunks: List[str] = None


@dataclass
class RegisterAccessIdentityEvent:
    name: str = "onRegisterAccessIdentity"
    identity: AccessIdentity = AccessIdentity()


@dataclass
class GetAccessIdentityEvent:
    name: str = "onGetAccessIdentity"
    resource: str = ""
    identity: AccessIdentity = AccessIdentity()


@dataclass
class AddDocumentsToVectorDBEvent:
    name: str = "onAddDocumentsToVectorDB"
    documents: List[Document] = None


@dataclass
class QueryVectorDBEvent:
    name: str = "onQueryVectorDB"
    query: VectorDBQuery = VectorDBQuery()
    vectorEmbeddings: List[VectorEmbedding] = None


@dataclass
class RetrieverFilterAccessibleFragmentsEvent:
    name: str = "onRetrieverFilterAccessibleFragments"
    fragments: List[DocumentFragment] = None


@dataclass
class RetrieverGetDocumentsForFragmentsEvent:
    name: str = "onRetrieverGetDocumentsForFragments"
    documents: List[Document] = None


@dataclass
class RetrieverProcessDocumentsEvent:
    name: str = "onRetrieverProcessDocuments"
    documents: List[Document] = None


@dataclass
class RetrieveDataEvent:
    name: str = "onRetrieveData"
    data: Any = None


@dataclass
class GetFragmentsEvent:
    name: str = "onGetFragments"
    fragments: List[DocumentFragment] = None


@dataclass
class RunCompletionEvent:
    name: str = "onRunCompletion"
    params: CompletionModelParams = CompletionModelParams()
    response: Any = None


@dataclass
class RunCompletionGenerationEvent:
    name: str = "onRunCompletionGeneration"
    params: LLMCompletionGeneratorParams = LLMCompletionGeneratorParams()
    response: Any = None


@dataclass
class GetRAGCompletionRetrievalQueryEvent:
    name: str = "onGetRAGCompletionRetrievalQuery"
    params: Any = None
    query: Any = None


# Define CallbackEvent union type
CallbackEvent = Union[
    LoadDocumentsSuccessEvent,
    LoadDocumentsErrorEvent,
    DataSourceTestConnectionSuccessEvent,
    DataSourceTestConnectionErrorEvent,
    ParseNextErrorEvent,
    ParseErrorEvent,
    ParseSuccessEvent,
    TransformDocumentsEvent,
    TransformDocumentEvent,
    ChunkTextEvent,
    RegisterAccessIdentityEvent,
    GetAccessIdentityEvent,
    AddDocumentsToVectorDBEvent,
    QueryVectorDBEvent,
    RetrieverFilterAccessibleFragmentsEvent,
    RetrieverGetDocumentsForFragmentsEvent,
    RetrieverProcessDocumentsEvent,
    RetrieveDataEvent,
    GetFragmentsEvent,
    RunCompletionEvent,
    RunCompletionGenerationEvent,
    GetRAGCompletionRetrievalQueryEvent,
]


# Define Callback class
class CallbackManager:
    runId: str
    callbacks: dict

    def __init__(self, runId: str, callbacks: dict) -> None:
        self.runId = runId
        self.callbacks = callbacks

    async def run_callbacks(self, event: CallbackEvent) -> None:
        if event.name in self.callbacks:
            for callback in self.callbacks[event.name]:
                await callback(event, self.runId)
        else:
            raise AssertionError("Unhandled event name")


# Define Traceable interface
class Traceable:
    callback_manager: CallbackManager = CallbackManager("", {})


Callback = Type[Callable[[CallbackEvent, str], Awaitable[None]]]


@dataclass
class CallbackMapping:
    onLoadDocumentsSuccess: Optional[List] = None
    onLoadDocumentsError: Optional[List] = None
    onDataSourceTestConnectionSuccess: Optional[List] = None
    onDataSourceTestConnectionError: Optional[List] = None
    onParseNextError: Optional[List] = None
    onParseError: Optional[List] = None
    onParseSuccess: Optional[List] = None
    onTransformDocuments: Optional[List] = None
    onTransformDocument: Optional[List] = None
    onChunkText: Optional[List] = None
    onRegisterAccessIdentity: Optional[List] = None
    onGetAccessIdentity: Optional[List] = None
    onAddDocumentToVectorDB: Optional[List] = None
    onQueryVectorDB: Optional[List] = None
    onRetrieverFilterAccessibleFragments: Optional[List] = None
    onRetrieverGetDocumentsForFragments: Optional[List] = None
    onRetrieverProcessDocuments: Optional[List] = None
    onRetrieveData: Optional[List] = None
    onGetFragments: Optional[List] = None
    onRunCompletion: Optional[List] = None
    onRunCompletionGeneration: Optional[List] = None
    onGetRAGCompletionRetrievalQuery: Optional[List] = None


# Define DEFAULT_CALLBACKS
DEFAULT_CALLBACKS = {"onLoadDocumentsSuccess": []}
