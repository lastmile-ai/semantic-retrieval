import { FinancialReportIdentity } from "./financialReportIdentity";

export class AdvisorIdentity implements FinancialReportIdentity {
  resource: "financial_data" = "financial_data";
  role: "user" | "admin" = "user";
}
