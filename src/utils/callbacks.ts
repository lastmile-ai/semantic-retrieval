import type { RawDocument } from "../document/document";

type DataSourceEventName = "onLoadDocumentsSuccess" | "onLoadDocumentsError";
type ParserEventName = "s1" | "s2";
// type CallbackEventName = DataSourceEventName | ParserEventName | TransformerEventName | IndexEventName | RetrieverEventName;
type CallbackEventName = DataSourceEventName | ParserEventName;


export type LoadDocumentsSuccessEvent = {
    name: "onLoadDocumentsSuccess";
    rawDocuments: RawDocument[];
}

export type LoadDocumentsErrorEvent = {
    name: "onLoadDocumentsError";
    message: string
}

type DataSourceEventData = LoadDocumentsSuccessEvent | LoadDocumentsErrorEvent;

type CallbackEvent = DataSourceEventData // | other stuff

type Callback<T extends CallbackEvent> = (event: T, rag_run_id: string) => Promise<void>;

// type CallbackMapping = { [key: string]: Callback[] }
interface CallbackMapping {
    onLoadDocumentsSuccess?: Callback<LoadDocumentsSuccessEvent>[];
    onLoadDocumentsError?: Callback<LoadDocumentsErrorEvent>[];
}

const DEFAULT_CALLBACKS: CallbackMapping = {
    onLoadDocumentsSuccess: []
}


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
        switch (event.name) {
            case "onLoadDocumentsSuccess":
                return await this.callback_helper(event, this.callbacks.onLoadDocumentsSuccess, DEFAULT_CALLBACKS.onLoadDocumentsSuccess);
            case "onLoadDocumentsError":
                return await this.callback_helper(event, this.callbacks.onLoadDocumentsError, DEFAULT_CALLBACKS.onLoadDocumentsError);
            default:
                console.log("default");
        }
    }

    private async callback_helper<T extends CallbackEvent>(event: T, userCallbacks?: Callback<T>[], defaultCallbacks?: Callback<T>[]) {
        const allCallbacks = (userCallbacks || []).concat(defaultCallbacks || []);
        for (const callback of allCallbacks) {
            // TODO return
            await callback(event, this.rag_run_id);
        }
    }
}

export { CallbackManager, CallbackMapping, CallbackEventName }