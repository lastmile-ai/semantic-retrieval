import { CallbackEvent, CallbackManager } from "./callbacks";
import * as util from "util";

export interface CallbackLogger {
  log: (message: string) => void;
  error: (message: string) => void;
}

export class LogEventCallbackManager extends CallbackManager {
  logger: CallbackLogger = console;

  constructor(runId: string, logger?: CallbackLogger) {
    super(runId, {});
    this.logger = logger ?? this.logger;
  }

  async runCallbacks(event: CallbackEvent) {
    // Use inspect instead of JSON.stringify to avoid circular reference errors
    const eventString = util.inspect(event);
    if (Object.prototype.hasOwnProperty.call(event, "error")) {
      this.logger.error(eventString);
    } else {
      this.logger.log(eventString);
    }
  }
}
