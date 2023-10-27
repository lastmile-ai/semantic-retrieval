import { FinancialReportIdentity } from "./financialReportIdentity";

export class AdminIdentity implements FinancialReportIdentity {
  resource: "financial_data";
  role: "admin";
}
