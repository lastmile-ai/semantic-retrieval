import { AccessIdentity } from "../../../../src/access-control/accessIdentity";

export interface FinancialReportIdentity extends AccessIdentity {
  resource: "financial_data";
  role: "user" | "admin";
  client?: string;
}
