import { FinancialReportIdentity } from "./financialReportIdentity";

export class AdvisorIdentity implements FinancialReportIdentity {
  resource: "financial_data";
  role: "user";
  client?: string;

  constructor(client: string) {
    this.client = client;
  }
}
