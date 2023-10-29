/* eslint-disable @typescript-eslint/no-explicit-any */

type CallbackEvent = any;

type Callback = (event: CallbackEvent, runId: string) => Promise<void>;

class CallbackManager {
  runId: string;
  callbacks: Callback[];

  constructor(runId: string, callbacks: Callback[]) {
    this.runId = runId;
    this.callbacks = callbacks;
  }

  async runCallbacks(event: CallbackEvent) {
    for (const callback of this.callbacks) {
      await callback(event, this.runId);
    }
  }
}

interface Traceable {
  callbackManager?: CallbackManager;
}

export { Traceable, CallbackManager, Callback, CallbackEvent };
