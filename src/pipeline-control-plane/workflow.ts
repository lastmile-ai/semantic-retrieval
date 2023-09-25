import { Attributable } from "../common/base";
import { JSONObject } from "../common/jsonTypes";
import { BlobIdentifier } from "../common/storage";
import { Pipeline } from "./pipeline";

/**
 * An instance of a pipeline execution run.
 */
export interface Workflow extends Attributable {
  workflowId: string;
  pipelineId: string;
  pipeline?: Pipeline;

  startTime: Date;
  finishTime?: Date;

  // The log of the workflow execution.
  transactionLog?: BlobIdentifier;

  // The current state of the workflow.
  currentState?: WorkflowState;
  startState?: WorkflowState;

  // Replay states of the workflow
  history: WorkflowState[];
}

export interface WorkflowState {
  workflowStateId: string;
  state: "pending" | "running" | "completed" | "failed";
  timestamp: Date;
  message?: string;
  data: JSONObject;

  previous?: WorkflowState;
  next?: WorkflowState;
}
