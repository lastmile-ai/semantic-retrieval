/* eslint-disable @typescript-eslint/no-explicit-any */
import { AccessIdentity } from "../access-control/accessIdentity";
import { VectorDBQuery } from "../data-store/vector-DBs/vectorDB";
import type {
  IngestedDocument,
  RawDocument,
  Document,
} from "../document/document";
import { VectorEmbedding } from "../transformation/embeddings/embeddings";

export type LoadDocumentsSuccessEvent = {
  name: "onLoadDocumentsSuccess";
  rawDocuments: RawDocument[];
};

export type LoadDocumentsErrorEvent = {
  name: "onLoadDocumentsError";
  error: Error;
};

export type DataSourceTestConnectionSuccessEvent = {
  name: "onDataSourceTestConnectionSuccess";
  code: number;
};

export type DataSourceTestConnectionErrorEvent = {
  name: "onDataSourceTestConnectionError";
  code: number;
  error: any;
};

export type ParseNextErrorEvent = {
  name: "onParseNextError";
  error: any;
};

export type ParseErrorEvent = {
  name: "onParseError";
  error: any;
};

export type ParseSuccessEvent = {
  name: "onParseSuccess";
  ingestedDocument: IngestedDocument;
};

export type TranformDocumentsEvent = {
  name: "onTransformDocuments";
  transformedDocuments: Document[];
  originalDocuments: Document[];
};

export type TransformDocumentEvent = {
  name: "onTransformDocument";
  originalDocument: Document;
  transformedDocument: Document;
};

export type ChunkTextEvent = {
  name: "onChunkText";
  chunks: string[];
};

export type RegisterAccessIdentityEvent = {
  name: "onRegisterAccessIdentity";
  identity: AccessIdentity;
};

export type GetAccessIdentityEvent = {
  name: "onGetAccessIdentity";
  resource: string;
  identity?: AccessIdentity;
};

export type AddDocumentsToVectorDBEvent = {
  name: "onAddDocumentsToVectorDB";
  documents: Document[];
};

export type QueryVectorDBEvent = {
  name: "onQueryVectorDB";
  query: VectorDBQuery;
  vectorEmbeddings: VectorEmbedding[];
};

type CallbackEvent =
  | LoadDocumentsSuccessEvent
  | LoadDocumentsErrorEvent
  | DataSourceTestConnectionSuccessEvent
  | DataSourceTestConnectionErrorEvent
  | ParseNextErrorEvent
  | ParseErrorEvent
  | ParseSuccessEvent
  | TranformDocumentsEvent
  | TransformDocumentEvent
  | ChunkTextEvent
  | RegisterAccessIdentityEvent
  | GetAccessIdentityEvent
  | AddDocumentsToVectorDBEvent
  | QueryVectorDBEvent;

type Callback<T extends CallbackEvent> = (
  event: T,
  runId: string
) => Promise<void>;

interface CallbackMapping {
  onLoadDocumentsSuccess?: Callback<LoadDocumentsSuccessEvent>[];
  onLoadDocumentsError?: Callback<LoadDocumentsErrorEvent>[];
  onDataSourceTestConnectionSuccess?: Callback<DataSourceTestConnectionSuccessEvent>[];
  onDataSourceTestConnectionError?: Callback<DataSourceTestConnectionErrorEvent>[];
  onParseNextError?: Callback<ParseNextErrorEvent>[];
  onParseError?: Callback<ParseErrorEvent>[];
  onParseSuccess?: Callback<ParseSuccessEvent>[];
  onTransformDocuments?: Callback<TranformDocumentsEvent>[];
  onTransformDocument?: Callback<TransformDocumentEvent>[];
  onChunkText?: Callback<ChunkTextEvent>[];
  onRegisterAccessIdentity?: Callback<RegisterAccessIdentityEvent>[];
  onGetAccessIdentity?: Callback<GetAccessIdentityEvent>[];
  onAddDocumentToVectorDB?: Callback<AddDocumentsToVectorDBEvent>[];
  onQueryVectorDB?: Callback<QueryVectorDBEvent>[];
}

const DEFAULT_CALLBACKS: CallbackMapping = {
  onLoadDocumentsSuccess: [],
};

class CallbackManager {
  runId: string;
  callbacks: CallbackMapping;

  constructor(runId: string, callbacks: CallbackMapping) {
    this.runId = runId;
    this.callbacks = callbacks;
  }

  async runCallbacks(event: CallbackEvent) {
    switch (event.name) {
      case "onLoadDocumentsSuccess":
        return await this.callback_helper(
          event,
          this.callbacks.onLoadDocumentsSuccess,
          DEFAULT_CALLBACKS.onLoadDocumentsSuccess
        );
      case "onLoadDocumentsError":
        return await this.callback_helper(
          event,
          this.callbacks.onLoadDocumentsError,
          DEFAULT_CALLBACKS.onLoadDocumentsError
        );
      case "onDataSourceTestConnectionSuccess":
        return await this.callback_helper(
          event,
          this.callbacks.onDataSourceTestConnectionSuccess,
          DEFAULT_CALLBACKS.onDataSourceTestConnectionSuccess
        );
      case "onDataSourceTestConnectionError":
        return await this.callback_helper(
          event,
          this.callbacks.onDataSourceTestConnectionError,
          DEFAULT_CALLBACKS.onDataSourceTestConnectionError
        );
      case "onParseNextError":
        return await this.callback_helper(
          event,
          this.callbacks.onParseNextError,
          DEFAULT_CALLBACKS.onParseNextError
        );
      case "onParseError":
        return await this.callback_helper(
          event,
          this.callbacks.onParseError,
          DEFAULT_CALLBACKS.onParseError
        );
      case "onParseSuccess":
        return await this.callback_helper(
          event,
          this.callbacks.onParseSuccess,
          DEFAULT_CALLBACKS.onParseSuccess
        );
      case "onTransformDocuments":
        return await this.callback_helper(
          event,
          this.callbacks.onTransformDocuments,
          DEFAULT_CALLBACKS.onTransformDocuments
        );
      case "onTransformDocument":
        return await this.callback_helper(
          event,
          this.callbacks.onTransformDocument,
          DEFAULT_CALLBACKS.onTransformDocument
        );
      case "onChunkText":
        return await this.callback_helper(
          event,
          this.callbacks.onChunkText,
          DEFAULT_CALLBACKS.onChunkText
        );
      case "onRegisterAccessIdentity":
        return await this.callback_helper(
          event,
          this.callbacks.onRegisterAccessIdentity,
          DEFAULT_CALLBACKS.onRegisterAccessIdentity
        );
      case "onGetAccessIdentity":
        return await this.callback_helper(
          event,
          this.callbacks.onGetAccessIdentity,
          DEFAULT_CALLBACKS.onGetAccessIdentity
        );
      case 'onAddDocumentsToVectorDB':
        return await this.callback_helper(
          event,
          this.callbacks.onAddDocumentToVectorDB,
          DEFAULT_CALLBACKS.onAddDocumentToVectorDB
        );
      case 'onQueryVectorDB':
        return await this.callback_helper(
          event,
          this.callbacks.onQueryVectorDB,
          DEFAULT_CALLBACKS.onQueryVectorDB
        );
      default:
        assertUnreachable(event);
    }
  }

  private async callback_helper<T extends CallbackEvent>(
    event: T,
    userCallbacks?: Callback<T>[],
    defaultCallbacks?: Callback<T>[]
  ) {
    const allCallbacks = (userCallbacks || []).concat(defaultCallbacks || []);
    for (const callback of allCallbacks) {
      // TODO return
      await callback(event, this.runId);
    }
  }
}

interface Traceable {
  callbackManager?: CallbackManager;
}

export { Traceable, CallbackManager, CallbackMapping };
