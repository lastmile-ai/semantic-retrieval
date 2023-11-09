import { AccessPassport } from "../../src";
import { AlwaysAllowAccessPolicy } from "../../src/access-control/policies/alwaysAllowAccessPolicy";
import { AlwaysDenyAccessPolicy } from "../../src/access-control/policies/alwaysDenyAccessPolicy";
import { CSVRetriever } from "../../src/retrieval/csvRetriever";

const accessPassport = new AccessPassport();
accessPassport.register({ resource: "*" });

describe("csvRetriever for retrieving structured data from CSV", () => {
  test("throws error if primary key column is missing", async () => {
    const retriever = new CSVRetriever({
      path: "../examples/example_data/financial_report/portfolios/sarmad_portfolio.csv",
      sourceAccessPolicies: [new AlwaysAllowAccessPolicy()],
    });

    await expect(
      retriever.retrieveData({
        query: {
          primaryKeyColumn: "DoesNotExist",
        },
        accessPassport,
      })
    ).rejects.toThrow(`Primary key column DoesNotExist missing from row`);
  });

  test("returns null if no access policies are set", async () => {
    const retriever = new CSVRetriever({
      path: "../examples/example_data/financial_report/portfolios/sarmad_portfolio.csv",
      sourceAccessPolicies: [],
    });

    const data = await retriever.retrieveData({
      query: {
        primaryKeyColumn: "Company",
      },
      accessPassport,
    });

    expect(data).toBe(null);
  });

  test("returns null if all access policies fail", async () => {
    const retriever = new CSVRetriever({
      path: "../examples/example_data/financial_report/portfolios/sarmad_portfolio.csv",
      sourceAccessPolicies: [new AlwaysDenyAccessPolicy()],
    });

    const data = await retriever.retrieveData({
      query: {
        primaryKeyColumn: "Company",
      },
      accessPassport,
    });

    expect(data).toBe(null);
  });

  test("retrieveData returns correct data from CSV", async () => {
    const retriever = new CSVRetriever<{
      [Company: string]: { Shares: number | null };
    }>({
      path: "../examples/example_data/financial_report/portfolios/sarmad_portfolio.csv",
      sourceAccessPolicies: [new AlwaysAllowAccessPolicy()],
    });

    const data = (await retriever.retrieveData({
      query: {
        primaryKeyColumn: "Company",
      },
      accessPassport,
    })) as {
      [Company: string]: { Shares: number | null };
    };

    expect(data["AAPL"].Shares).toEqual(20);
    expect(data["MSFT"].Shares).toEqual(null);
    expect(data["AMZN"].Shares).toEqual(30);
    expect(data["NVDA"].Shares).toEqual(100);
    expect(data["TSLA"].Shares).toEqual(null);
    expect(data["GOOG"].Shares).toEqual(null);
    expect(data["BRK.B"].Shares).toEqual(null);
    expect(data["META"].Shares).toEqual(null);
    expect(data["UNH"].Shares).toEqual(30);
    expect(data["XOM"].Shares).toEqual(null);
    expect(data["LLY"].Shares).toEqual(null);
    expect(data["JPM"].Shares).toEqual(null);
    expect(data["JNJ"].Shares).toEqual(100);
    expect(data["V"].Shares).toEqual(null);
    expect(data["PG"].Shares).toEqual(null);
    expect(data["MA"].Shares).toEqual(null);
    expect(data["AVGO"].Shares).toEqual(null);
    expect(data["HD"].Shares).toEqual(null);
    expect(data["CVX"].Shares).toEqual(null);
    expect(data["MRK"].Shares).toEqual(40);
    expect(data["ABBV"].Shares).toEqual(null);
    expect(data["COST"].Shares).toEqual(null);
    expect(data["PEP"].Shares).toEqual(200);
    expect(data["ADBE"].Shares).toEqual(null);
  });
});
