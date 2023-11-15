import { DocumentRetriever } from "../retrieval/documentRetriever";
import {
  BaseRetrieverQueryParams,
  RetrieverResponse,
} from "../retrieval/retriever";

// Interface

// Types
export interface RetrievalEvaluationDataset<E> {
  relevantDataByQuery: retrievalQueryExpectedResultPair<E>[];
}

export interface RetrievalMetric<R, E> {
  fn: (retrievedData: R, relevantData: E) => number;
  name: string;
}

/**
 * Perform evaluation of the retrievers with provided data and metrics
 * @param retrievers DocumentRetrievers to perform evaluation on
 * @param data Dataset of retrieval queries and expected results of type E
 * @param metrics Array of RetrievalMetrics to evaluate
 * @returns
 */
export async function evaluateRetrievers<R extends DocumentRetriever, E>(
  retrievers: R[],
  data: RetrievalEvaluationDataset<E>,
  metrics: RetrievalMetric<RetrieverResponse<R>, E>[]
) {
  const results: { [metricName: string]: number }[] = [];

  for (const retriever of retrievers) {
    const record: { [metricName: string]: number } = {};
    for (const metric of metrics) {
      record[metric.name] = await averageMetricValue(retriever, metric, data);
    }
    results.push(record);
  }

  return results;
}

// Internal convenience types

type retrievalQueryExpectedResultPair<E> = [BaseRetrieverQueryParams, E];

// Common metrics

export function calculateRecall<T>(
  retrievedIds: Set<T>,
  relevantIds: Set<T>
): number {
  const intersection = setIntersect(retrievedIds, relevantIds);
  return intersection.size / relevantIds.size;
}

export function calculatePrecision<T>(
  retrievedIds: Set<T>,
  relevantIds: Set<T>
): number {
  const intersection = setIntersect(retrievedIds, relevantIds);
  return intersection.size / retrievedIds.size;
}

// Utility functions

function setIntersect<T>(a: Set<T>, b: Set<T>): Set<T> {
  const intersection: Set<T> = new Set();
  for (const elem of a) {
    if (b.has(elem)) {
      intersection.add(elem);
    }
  }
  return intersection;
}

async function averageMetricValue<R extends DocumentRetriever, E>(
  retriever: R,
  metric: RetrievalMetric<RetrieverResponse<R>, E>,
  data: RetrievalEvaluationDataset<E>
): Promise<number> {
  let total = 0;
  const dataList: retrievalQueryExpectedResultPair<E>[] =
    data.relevantDataByQuery;
  if (data.relevantDataByQuery.length === 0) {
    throw new Error("No data in evaluation dataset");
  }

  for (let i = 0; i < dataList.length; i++) {
    const qrp = dataList[i];
    const [query, relevantData] = qrp;
    const retrievedData = (await retriever.retrieveData(
      query
    )) as RetrieverResponse<R>;

    total += await metric.fn(retrievedData, relevantData);
  }
  return total / data.relevantDataByQuery.length;
}
