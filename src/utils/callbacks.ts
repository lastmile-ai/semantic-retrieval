// type CallbackEventName = DataSourceEventName | ParserEventName | TransformerEventName | IndexEventName | RetrieverEventName;
type DataSourceEventName = "onFileLoaded" | "onFileLoadError";
type ParserEventName = "s1" | "s2";
type CallbackEventName = DataSourceEventName | ParserEventName;

type LoadDocumentErrorEvent = {
  name: "onFileLoadError";
  data: number;
};

import type { RawDocument } from "../document/document";

// Remove
export type LoadDocumentsSuccessEvent = {
  name: "onLoadDocumentsSuccess";
  rawDocuments: RawDocument[];
};

// Remove
export type LoadDocumentsErrorEvent = {
  name: "onLoadDocumentsError";
  message: string;
};

// Remove
type DataSourceEvent = LoadDocumentsSuccessEvent | LoadDocumentErrorEvent;

// Remove
type CallbackEvent = DataSourceEvent;

export type Event<T> = {
  name: string;
  data: T;
};

export type Callback<T> = (
  event: Event<T>,
  rag_run_id: string
) => Promise<void>;

export interface ILoadDocumentsSuccessEvent extends Event<RawDocument[]> {
  name: "onLoadDocumentsSuccess";
  // data is type RawDocument[] (previously called rawDocument)
}

export type ILoadDocumentsSuccessCallback = Callback<
  ILoadDocumentsSuccessEvent["data"]
>;

export interface ILoadDocumentsErrorEvent extends Event<string> {
  name: "onLoadDocumentsError";
  // data is type string (previously called message)
}

export type ILoadDocumentsErrorCallback = Callback<
  ILoadDocumentsErrorEvent["data"]
>;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type CallbackMapping = { [key: string]: Callback<any>[] };

// const DEFAULT_CALLBACKS: CallbackMapping = {
//     "onFileLoaded": []
// }

export class CallbackManager {
  // Properties
  rag_run_id: string;
  callbacks: CallbackMapping;

  // Constructor
  constructor(rag_run_id: string, callbacks: CallbackMapping) {
    this.rag_run_id = rag_run_id;
    this.callbacks = callbacks;
  }

  async runCallbacks(event: Event<unknown>) {
    if (event.name in this.callbacks) {
      const callbacks = this.callbacks[event.name];
      for (const callback of callbacks) {
        // TODO: return
        await callback(event, this.rag_run_id);
      }
    }
  }
}
