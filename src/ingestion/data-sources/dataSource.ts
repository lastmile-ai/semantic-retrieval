import { JSONObject } from "../../common/jsonTypes";
import { RawDocument } from "../../document/document";
import {
  CallbackManager,
  DataSourceTestConnectionErrorEvent,
  LoadDocumentsErrorEvent,
} from "../../utils/callbacks";

export interface Authentication {
  authToken?: string;

  // TODO: saqadri - add any other kinds of authentication methods here.
}

/**
 * Defines how to connect to a data source.
 */
export type ConnectionInfo = JSONObject & {
  endpoint?: string;
  auth?: Authentication;
};

/**
 * The base interface for all data sources to load documents from.
 */
export interface DataSource {
  // The name of the data source.
  name: string;

  // Connection information (including authentication) for the data source.
  connectionInfo?: ConnectionInfo;

  // Any JSON-serializable metadata associated with the data source.
  metadata?: { [key: string]: string };
  // A general property bag associated with this object.
  attributes?: { [key: string]: string };

  /**
   * Tests the connection to the data source. HTTP status codes are used to indicate success or failure.
   */
  testConnection(): Promise<number>;

  /**
   * Loads documents from the data source that matches the specified properties.
   * @param filters The filters to apply to the documents. Can include/exclude things like document IDs, recursive traversal, file extensions, etc.
   * @param limit The maximum number of documents to load.
   */
  loadDocuments(filters: JSONObject, limit?: number): Promise<RawDocument[]>;
}

export class GoogleDrive implements DataSource {
  name: string = "GDrive";
  connectionInfo?: ConnectionInfo | undefined;
  metadata?: { [key: string]: string } | undefined;
  attributes?: { [key: string]: string } | undefined;
  callbackManager?: CallbackManager;

  constructor(callbackManager?: CallbackManager) {
    this.callbackManager = callbackManager ?? undefined;
  }

  testConnection(): Promise<number> {
    const err = new Error("Method not implemented.");
    const event: DataSourceTestConnectionErrorEvent = {
      name: "onDataSourceTestConnectionError",
      code: -1,
      error: err,
    };
    this.callbackManager?.runCallbacks(event);
    throw err;
  }
  loadDocuments(
    _filters: JSONObject,
    _limit?: number | undefined
  ): Promise<RawDocument[]> {
    const err = new Error("Method not implemented.");
    const event: LoadDocumentsErrorEvent = {
      name: "onLoadDocumentsError",
      error: err,
    };
    this.callbackManager?.runCallbacks(event);
    throw err;
  }
}

export class OneDrive implements DataSource {
  name: string = "OneDrive";
  connectionInfo?: ConnectionInfo | undefined;
  metadata?: { [key: string]: string } | undefined;
  attributes?: { [key: string]: string } | undefined;
  callbackManager?: CallbackManager;

  constructor(callbackManager?: CallbackManager) {
    this.callbackManager = callbackManager ?? undefined;
  }

  testConnection(): Promise<number> {
    const err = new Error("Method not implemented.");
    const event: DataSourceTestConnectionErrorEvent = {
      name: "onDataSourceTestConnectionError",
      code: -1,
      error: err,
    };
    this.callbackManager?.runCallbacks(event);
    throw err;
  }
  loadDocuments(
    _filters: JSONObject,
    _limit?: number | undefined
  ): Promise<RawDocument[]> {
    const err = new Error("Method not implemented.");
    const event: LoadDocumentsErrorEvent = {
      name: "onLoadDocumentsError",
      error: err,
    };
    this.callbackManager?.runCallbacks(event);
    throw err;
  }
}
