import { AccessIdentity } from "../../../../../src/access-control/accessIdentity";
import { ResourceAccessPolicy } from "../../../../../src/access-control/resourceAccessPolicy";
import { Document } from "../../../../../src/document/document";
import { isFinancialReportIdentity } from "./financialReportIdentity";

export const RESOURCE = "financial_data";

export class SecretReportAccessPolicy implements ResourceAccessPolicy {
  policy = "SecretReportAccessPolicy";
  resource = RESOURCE;

  async testDocumentReadPermission(
    _document: Document,
    requestor?: AccessIdentity
  ) {
    if (requestor && isFinancialReportIdentity(requestor)) {
      return requestor.role === "admin";
    }
    return false;
  }

  async testPolicyPermission(_requestor: AccessIdentity) {
    return false; // Not needed for this example
  }
}
