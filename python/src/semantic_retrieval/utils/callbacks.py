from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Sequence, Union


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
from typing import Optional, Any, List


@dataclass
class LoadDocumentsSuccessEvent:
    name: str = "onLoadDocumentsSuccess"
    rawDocuments: Optional[List[Any]] = None


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
    transformedDocuments: Optional[List[Document]] = None
    originalDocuments: Optional[List[Document]] = None


@dataclass
class TransformDocumentEvent:
    name: str = "onTransformDocument"
    originalDocument: Document = Document()
    transformedDocument: Document = Document()


@dataclass
class ChunkTextEvent:
    name: str = "onChunkText"
    chunks: Optional[List[str]] = None


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
    documents: Optional[List[Document]] = None


@dataclass
class QueryVectorDBEvent:
    name: str = "onQueryVectorDB"
    query: VectorDBQuery = VectorDBQuery()
    vectorEmbeddings: Optional[List[VectorEmbedding]] = None


@dataclass
class RetrieverFilterAccessibleFragmentsEvent:
    name: str = "onRetrieverFilterAccessibleFragments"
    fragments: Optional[List[DocumentFragment]] = None


@dataclass
class RetrieverGetDocumentsForFragmentsEvent:
    name: str = "onRetrieverGetDocumentsForFragments"
    documents: Optional[List[Document]] = None


@dataclass
class RetrieverProcessDocumentsEvent:
    name: str = "onRetrieverProcessDocuments"
    documents: Optional[List[Document]] = None


@dataclass
class RetrieveDataEvent:
    name: str = "onRetrieveData"
    data: Any = None


@dataclass
class GetFragmentsEvent:
    name: str = "onGetFragments"
    fragments: Optional[List[DocumentFragment]] = None


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

Callback = Callable[[CallbackEvent, str], Awaitable[None]]


@dataclass
class CallbackManager:
    runId: str
    callbacks: dict[str, Sequence[Callback]]

    async def run_callbacks(self, event: CallbackEvent) -> None:
        # TODO
        pass


# Define Traceable interface
class Traceable:
    callback_manager: CallbackManager = CallbackManager("", {})


@dataclass
class CallbackMapping:
    onLoadDocumentsSuccess: Optional[List[Any]] = None
    onLoadDocumentsError: Optional[List[Any]] = None
    onDataSourceTestConnectionSuccess: Optional[List[Any]] = None
    onDataSourceTestConnectionError: Optional[List[Any]] = None
    onParseNextError: Optional[List[Any]] = None
    onParseError: Optional[List[Any]] = None
    onParseSuccess: Optional[List[Any]] = None
    onTransformDocuments: Optional[List[Any]] = None
    onTransformDocument: Optional[List[Any]] = None
    onChunkText: Optional[List[Any]] = None
    onRegisterAccessIdentity: Optional[List[Any]] = None
    onGetAccessIdentity: Optional[List[Any]] = None
    onAddDocumentToVectorDB: Optional[List[Any]] = None
    onQueryVectorDB: Optional[List[Any]] = None
    onRetrieverFilterAccessibleFragments: Optional[List[Any]] = None
    onRetrieverGetDocumentsForFragments: Optional[List[Any]] = None
    onRetrieverProcessDocuments: Optional[List[Any]] = None
    onRetrieveData: Optional[List[Any]] = None
    onGetFragments: Optional[List[Any]] = None
    onRunCompletion: Optional[List[Any]] = None
    onRunCompletionGeneration: Optional[List[Any]] = None
    onGetRAGCompletionRetrievalQuery: Optional[List[Any]] = None


# Define DEFAULT_CALLBACKS
DEFAULT_CALLBACKS = {"onLoadDocumentsSuccess": []}
