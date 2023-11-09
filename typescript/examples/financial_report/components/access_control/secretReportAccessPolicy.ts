import { AccessIdentity } from "../../../../../typescript/src/access-control/accessIdentity";
import { ResourceAccessPolicy } from "../../../../../typescript/src/access-control/resourceAccessPolicy";
import { Document } from "../../../../../typescript/src/document/document";
import { DocumentMetadataDB } from "../../../../src/document/metadata/documentMetadataDB";
import { isFinancialReportIdentity } from "./financialReportIdentity";

export const RESOURCE = "financial_data";

export class SecretReportAccessPolicy implements ResourceAccessPolicy {
  policy = "SecretReportAccessPolicy";
  resource = RESOURCE;

  async testDocumentReadPermission(
    _document: Document,
    _metadataDB: DocumentMetadataDB,
    requestor?: AccessIdentity
  ) {
    if (requestor && isFinancialReportIdentity(requestor)) {
      return requestor.role === "admin";
    }
    return false;
  }

  async testPolicyPermission(requestor: AccessIdentity) {
    if (!(requestor && isFinancialReportIdentity(requestor))) {
      return false;
    }

    return requestor.role === "admin" ? ["read", "write"] : false;
  }
}
