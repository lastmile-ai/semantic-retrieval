import type { RawDocument } from "../document/document";

export type LoadDocumentsSuccessEvent = {
  name: "onLoadDocumentsSuccess";
  rawDocuments: RawDocument[];
};

export type LoadDocumentsErrorEvent = {
  name: "onLoadDocumentsError";
  message: string;
};

export type DataSourceTestConnectionSuccessEvent = {
  name: "onDataSourceTestConnectionSuccess";
  code: number;
};

// Same as above but for error
export type DataSourceTestConnectionErrorEvent = {
  name: "onDataSourceTestConnectionError";
  code: number;
};

type DataSourceEventData =
  | LoadDocumentsSuccessEvent
  | LoadDocumentsErrorEvent
  | DataSourceTestConnectionSuccessEvent
  | DataSourceTestConnectionErrorEvent;

type CallbackEvent = DataSourceEventData; // | other stuff

type Callback<T extends CallbackEvent> = (
  event: T,
  runId: string
) => Promise<void>;

// type CallbackMapping = { [key: string]: Callback[] }
interface CallbackMapping {
  onLoadDocumentsSuccess?: Callback<LoadDocumentsSuccessEvent>[];
  onLoadDocumentsError?: Callback<LoadDocumentsErrorEvent>[];
  // 2 more cases, for testConnection
  onDataSourceTestConnectionSuccess?: Callback<DataSourceTestConnectionSuccessEvent>[];
  onDataSourceTestConnectionError?: Callback<DataSourceTestConnectionErrorEvent>[];
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
