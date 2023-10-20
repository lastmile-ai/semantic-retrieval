import type { RawDocument } from "../document/document";

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

type CallbackEvent =
  | LoadDocumentsSuccessEvent
  | LoadDocumentsErrorEvent
  | DataSourceTestConnectionSuccessEvent
  | DataSourceTestConnectionErrorEvent;

// type CallbackEvent = DataSourceEventData; // | other stuff

type Callback<T extends CallbackEvent> = (
  event: T,
  runId: string
) => Promise<void>;

interface CallbackMapping {
  onLoadDocumentsSuccess?: Callback<LoadDocumentsSuccessEvent>[];
  onLoadDocumentsError?: Callback<LoadDocumentsErrorEvent>[];
  onDataSourceTestConnectionSuccess?: Callback<DataSourceTestConnectionSuccessEvent>[];
  onDataSourceTestConnectionError?: Callback<DataSourceTestConnectionErrorEvent>[];
  // 2 more cases, for GoogleDrive
}

const DEFAULT_CALLBACKS: CallbackMapping = {
  onLoadDocumentsSuccess: [],
};

class CallbackManager {
  // Properties
  runId: string;
  callbacks: CallbackMapping;

  // Constructor
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
      // 2 more cases, for testConnection
      case "onDataSourceTestConnectionSuccess":
        return await this.callback_helper(
          event,
          this.callbacks.onDataSourceTestConnectionSuccess,
          DEFAULT_CALLBACKS.onDataSourceTestConnectionSuccess
        );
      // same as above but for error
      case "onDataSourceTestConnectionError":
        return await this.callback_helper(
          event,
          this.callbacks.onDataSourceTestConnectionError,
          DEFAULT_CALLBACKS.onDataSourceTestConnectionError
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

export { CallbackManager, CallbackMapping };
