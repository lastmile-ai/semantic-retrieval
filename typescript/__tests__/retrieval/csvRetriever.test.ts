import { AccessPassport } from "../../src";
import { AlwaysAllowAccessPolicy } from "../../src/access-control/policies/alwaysAllowAccessPolicy";
import { AlwaysDenyAccessPolicy } from "../../src/access-control/policies/alwaysDenyAccessPolicy";
import { CSVRetriever } from "../../src/retrieval/csvRetriever";
import * as path from "path";

const accessPassport = new AccessPassport();
accessPassport.register({ resource: "*" });

const testPortfolioPath = path.resolve(
  "__tests__/__mocks__/test_data/portfolio.csv"
);

describe("csvRetriever for retrieving structured data from CSV", () => {
  test("throws error if primary key column is missing", async () => {
    const retriever = new CSVRetriever({
      path: testPortfolioPath,
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
      path: testPortfolioPath,
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
      path: testPortfolioPath,
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
      path: testPortfolioPath,
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
    expect(data["AMZN"].Shares).toEqual(30);
    expect(data["COST"].Shares).toEqual(null);
    expect(data["PEP"].Shares).toEqual(200);
    expect(data["ADBE"].Shares).toEqual(null);
  });
});
