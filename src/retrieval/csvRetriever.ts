import { JSONObject } from "../common/jsonTypes";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";
import fs from "fs/promises";
import Papa from "papaparse";

export type CSVRetrieverQuery = {
  filePath: string;
  // For now, assume a single primary key column
  primaryKeyColumn: string;
};

/**
 * Retrieve structured data from a CSV file as a JSON object for use in completion generation.
 */
export class CSVRetriever<R extends JSONObject> extends BaseRetriever<
  R,
  CSVRetrieverQuery
> {
  /**
   * Get the data relevant to the given query and which the current identity can access.
   * @param params The retriever query params to use for the query.
   * @returns A promise that resolves to the retrieved data.
   */
  async retrieveData(
    params: BaseRetrieverQueryParams<CSVRetrieverQuery>
  ): Promise<R> {
    const csvString = await (
      await fs.readFile(params.query.filePath)
    ).toString();
    const rows = await Papa.parse<R>(csvString, {
      header: true,
      dynamicTyping: true,
    }).data;

    const retrievedData: JSONObject = {};

    for (const row of rows) {
      const { [params.query.primaryKeyColumn]: key, ...restRow } = row;
      if (key == null) {
        throw new Error(
          `Primary key column ${params.query.primaryKeyColumn} missing from row`
        );
      }
      const keyVal = typeof key === "string" ? key : JSON.stringify(key);
      retrievedData[keyVal] = restRow;
    }

    return retrievedData as R;
  }
}
