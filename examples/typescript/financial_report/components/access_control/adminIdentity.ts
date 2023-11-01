import { FinancialReportIdentity } from "./financialReportIdentity";

export class AdminIdentity implements FinancialReportIdentity {
  resource: "financial_data" = "financial_data";
  role: "user" | "admin" = "admin";
}
