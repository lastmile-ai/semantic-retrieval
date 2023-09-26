import { JSONObject } from "../../common/jsonTypes.js";
import { Document } from "../../document/document.js";
import { AccessIdentity } from "../accessIdentity.js";
import { ResourceAccessPolicy } from "../resourceAccessPolicy.js";

export class AlwaysAllowAccessPolicy implements ResourceAccessPolicy {
  policy: string = "AlwaysAllowAccessPolicy";
  policyJSON: JSONObject = {};

  async testDocumentReadPermission(
    _document: Document,
    _requestor?: AccessIdentity,
  ): Promise<boolean> {
    return true;
  }

  async testPolicyPermission(
    _requestor: AccessIdentity,
  ): Promise<string[] | boolean> {
    return true;
  }
}
