import { JSONObject } from "../common/jsonTypes";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";
import fs from "fs/promises";
import Papa from "papaparse";

export type CSVRetrieverQuery = {
  // For now, assume a single primary key column
  primaryKeyColumn: string;
};

export interface CSVRetrieverQueryParams
  extends BaseRetrieverQueryParams<CSVRetrieverQuery> {}

/**
 * Retrieve structured data from a CSV file as a JSON object for use in completion generation.
 */
export class CSVRetriever<R extends JSONObject> extends BaseRetriever {
  filePath: string;

  constructor(filePath: string) {
    super();
    this.filePath = filePath;
  }
  /**
   * Get the data relevant to the given query and which the current identity can access.
   * @param params The retriever query params to use for the query.
   * @returns A promise that resolves to the retrieved data.
   */
  async retrieveData(params: CSVRetrieverQueryParams): Promise<R> {
    const csvString = await (await fs.readFile(this.filePath)).toString();
    const rows = await Papa.parse<R>(csvString, {
      header: true,
      dynamicTyping: true,
    }).data;

    const retrievedData: JSONObject = {};
    const primaryKeyColumn = params.query.primaryKeyColumn;

    for (const row of rows) {
      const { [primaryKeyColumn]: key, ...restRow } = row;
      if (key == null) {
        if (restRow != null) {
          throw new Error(
            `Primary key column ${primaryKeyColumn} missing from row`
          );
        }
        continue; // ignore completely empty rows
      }

      const keyVal = typeof key === "string" ? key : JSON.stringify(key);
      retrievedData[keyVal] = restRow;
    }

    return retrievedData as R;
  }
}
