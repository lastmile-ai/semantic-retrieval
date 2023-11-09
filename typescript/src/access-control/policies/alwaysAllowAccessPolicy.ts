import { JSONObject } from "../../common/jsonTypes";
import { Document } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import { AccessIdentity } from "../accessIdentity";
import { ResourceAccessPolicy } from "../resourceAccessPolicy";

export class AlwaysAllowAccessPolicy implements ResourceAccessPolicy {
  policy: string = "AlwaysAllowAccessPolicy";
  policyJSON: JSONObject = {};

  async testDocumentReadPermission(
    _document: Document,
    _metadataDB: DocumentMetadataDB,
    _requestor?: AccessIdentity
  ): Promise<boolean> {
    return true;
  }

  async testPolicyPermission(_requestor: AccessIdentity) {
    return ["*", "read", "write"];
  }
}
