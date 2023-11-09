import { JSONObject } from "../../common/jsonTypes";
import { Document } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import { AccessIdentity } from "../accessIdentity";
import { GLOBAL_RESOURCE, ResourceAccessPolicy } from "../resourceAccessPolicy";

export class AlwaysDenyAccessPolicy implements ResourceAccessPolicy {
  resource: string = GLOBAL_RESOURCE;
  policy: string = "AlwaysDenyAccessPolicy";
  policyJSON: JSONObject = {};

  async testDocumentReadPermission(
    _document: Document,
    _metadataDB: DocumentMetadataDB,
    _requestor?: AccessIdentity
  ): Promise<boolean> {
    return false;
  }

  async testPolicyPermission(_requestor: AccessIdentity) {
    return false;
  }
}
