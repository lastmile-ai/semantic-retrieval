import { AccessIdentity } from "../../../../../src/access-control/accessIdentity";

export interface FinancialReportIdentity extends AccessIdentity {
  resource: "financial_data";
  role: "user" | "admin";
}

export function isFinancialReportIdentity(
  identity: AccessIdentity
): identity is FinancialReportIdentity {
  return identity.resource === "financial_data";
}
