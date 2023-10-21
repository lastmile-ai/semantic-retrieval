import type { IngestedDocument, RawDocument } from "../document/document";

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

type CallbackEvent =
  | LoadDocumentsSuccessEvent
  | LoadDocumentsErrorEvent
  | DataSourceTestConnectionSuccessEvent
  | DataSourceTestConnectionErrorEvent
  | ParseNextErrorEvent
  | ParseErrorEvent
  | ParseSuccessEvent;

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
