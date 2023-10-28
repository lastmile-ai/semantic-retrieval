import { CSVRetriever } from "../../src/retrieval/csvRetriever";

describe("csvRetriever for retrieving structured data from CSV", () => {
  test("throws error if primary key column is missing", async () => {
    const retriever = new CSVRetriever(
      "examples/example_data/financial_report/portfolios/client_a_portfolio.csv"
    );
    await expect(
      retriever.retrieveData({
        query: {
          primaryKeyColumn: "DoesNotExist",
        },
      })
    ).rejects.toThrowError(`Primary key column DoesNotExist missing from row`);
  });

  test("retrieveData returns correct data from CSV", async () => {
    const retriever = new CSVRetriever<{
      [Company: string]: { Shares: number | null };
    }>(
      "examples/example_data/financial_report/portfolios/client_a_portfolio.csv"
    );

    const data = await retriever.retrieveData({
      query: {
        primaryKeyColumn: "Company",
      },
    });

    expect(data["AAPL"].Shares).toEqual(20);
    expect(data["MSFT"].Shares).toEqual(null);
    expect(data["AMZN"].Shares).toEqual(30);
    expect(data["NVDA"].Shares).toEqual(100);
    expect(data["GOOGL"].Shares).toEqual(null);
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
