import { Attributable, Identifiable } from "../common/base";
import { DataSource } from "../ingestion/data-sources/dataSource";
import { Workflow } from "./workflow";

// TODO: saqadri -- this needs more definition
export interface Pipeline extends Attributable {
  pipelineId: string;
  config: PipelineConfig;

  latestWorkflow?: Workflow;
  workflows: Workflow[];

  start(): Promise<Workflow>;
  stop(): Promise<Workflow>;
  cancel(): Promise<Workflow>;
}

export interface PipelineConfig extends Attributable, Identifiable {
  dataSources: DataSource[];
  schedule?: Schedule;
}

export interface Schedule {
  cron: string;
  timezone?: string;
}
