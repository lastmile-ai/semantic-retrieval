import { AccessPassport, ResourceAccessPolicy } from "../access-control";
import { JSONObject } from "../common/jsonTypes";
import { RetrieveDataEvent } from "../utils/callbacks";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";
import fs from "fs/promises";
import Papa from "papaparse";

export type CSVRetrieverQuery = {
  // For now, assume a single primary key column
  primaryKeyColumn: string;
};

export type CSVRetrieverConfig = {
  path: string;
  sourceAccessPolicies: ResourceAccessPolicy[];
};

export interface CSVRetrieverQueryParams
  extends BaseRetrieverQueryParams<CSVRetrieverQuery> {
  accessPassport: AccessPassport;
}

/**
 * Retrieve structured data from a CSV file as a JSON object for use in completion generation.
 */
export class CSVRetriever<R extends JSONObject> extends BaseRetriever {
  filePath: string;
  sourceAccessPolicies: ResourceAccessPolicy[];

  constructor(config: CSVRetrieverConfig) {
    super();
    this.filePath = config.path;
    this.sourceAccessPolicies = config.sourceAccessPolicies;
  }

  private async checkSourceAccess(
    params: CSVRetrieverQueryParams
  ): Promise<{ canAccess: boolean; policyFailed?: ResourceAccessPolicy }> {
    // For now, just check access at the source (file) level; row-level policies can be added
    // later if needed
    const policyChecks = await Promise.all(
      this.sourceAccessPolicies.map(async (policy) => {
        const identity = params.accessPassport.getIdentity(policy.resource);
        return {
          policy,
          check: identity && (await policy.testPolicyPermission(identity)),
        };
      })
    );

    // Default to hidden content. If no policies, or any policy fails, return null.
    if (policyChecks.length === 0) {
      return { canAccess: false };
    }

    for (const { policy, check } of policyChecks) {
      if (!check) {
        return { canAccess: false, policyFailed: policy };
      }

      if (Array.isArray(check)) {
        if (check.includes("read") || check.includes("*")) {
          continue;
        }
      }

      return { canAccess: false, policyFailed: policy };
    }

    return { canAccess: true };
  }

  /**
   * Get the data relevant to the given query and which the current identity can access.
   * If the current identity does not have access to the data, return null.
   * @param params The retriever query params to use for the query.
   * @returns A promise that resolves to the retrieved data, or null if no data accessible.
   */
  async retrieveData(params: CSVRetrieverQueryParams): Promise<R | null> {
    const accessCheck = await this.checkSourceAccess(params);
    if (!accessCheck.canAccess) {
      await this.callbackManager?.runCallbacks({
        name: "onRetrieverSourceAccessPolicyCheckFailed",
        params,
        policy: accessCheck.policyFailed ?? null,
      });
      return null;
    }

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

    const event: RetrieveDataEvent = {
      name: "onRetrieveData",
      params,
      data: retrievedData,
    };

    await this.callbackManager?.runCallbacks(event);

    return retrievedData as R;
  }
}
