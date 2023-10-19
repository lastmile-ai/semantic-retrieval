// type CallbackEventName = DataSourceEventName | ParserEventName | TransformerEventName | IndexEventName | RetrieverEventName;
type DataSourceEventName = "onFileLoaded" | "onFileLoadError";
type ParserEventName = "s1" | "s2";
type CallbackEventName = DataSourceEventName | ParserEventName;


type LoadDocumentErrorEvent = {
    name: "onFileLoadError";
    data: number;
}

import type { RawDocument } from "../document/document";

export type LoadDocumentsSuccessEvent = {
    name: "onLoadDocumentsSuccess";
    rawDocuments: RawDocument[];
  }

export type LoadDocumentsErrorEvent= {
    name: "onLoadDocumentsError";
    message: string
}

type DataSourceEvent = LoadDocumentsSuccessEvent | LoadDocumentErrorEvent;

type CallbackEvent = DataSourceEvent

type Callback = (event: CallbackEvent, rag_run_id: string) => Promise<void>;

type CallbackMapping = { [key: string]: Callback[] }


// const DEFAULT_CALLBACKS: CallbackMapping = {
//     "onFileLoaded": []
// }


class CallbackManager {
    // Properties
    rag_run_id: string;
    callbacks: CallbackMapping;

    // Constructor
    constructor(rag_run_id: string, callbacks: CallbackMapping) {
        this.rag_run_id = rag_run_id;
        this.callbacks = callbacks;
    }

    async runCallbacks(event: CallbackEvent) {
        if (event.name in this.callbacks) {
            const callbacks = this.callbacks[event.name];
            for (const callback of callbacks) {
                // TODO: return
                await callback(event, this.rag_run_id);
            }
        }
    }
}

export { CallbackManager, CallbackMapping, CallbackEventName }