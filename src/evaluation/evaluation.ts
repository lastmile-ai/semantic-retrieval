import {
  BaseRetriever,
  BaseRetrieverQueryParams,
} from "../retrieval/retriever";

// Interface

// Types
export interface RetrievalEvaluationDataset<E, Q> {
  relevantDataByQuery: retrievalQueryExpectedResultPair<E, Q>[];
}

export interface RetrievalMetric<R, E> {
  fn: (retrievedData: R, relevantData: E) => number;
  name: string;
}

// Retrieval eval function
/* 
R: returned data type from retriever
Q: query type
E: expected data type

Example: R = Document[], Q = VectorDBQuery, E = Fragment[]
*/
export async function evaluateRetrievers<R, Q, E>(
  retrievers: BaseRetriever<R, Q>[],
  data: RetrievalEvaluationDataset<E, Q>,
  metrics: RetrievalMetric<R, E>[]
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

type retrievalQueryExpectedResultPair<E, Q> = [BaseRetrieverQueryParams<Q>, E];

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

async function averageMetricValue<R, Q, E>(
  retriever: BaseRetriever<R, Q>,
  metric: RetrievalMetric<R, E>,
  data: RetrievalEvaluationDataset<E, Q>
): Promise<number> {
  let total = 0;
  const dataList: retrievalQueryExpectedResultPair<E, Q>[] =
    data.relevantDataByQuery;
  if (data.relevantDataByQuery.length === 0) {
    throw new Error("No data in evaluation dataset");
  }

  for (let i = 0; i < dataList.length; i++) {
    const qrp = dataList[i];
    const [query, relevantData] = qrp;
    const retrievedData = await retriever.retrieveData(query);

    total += await metric.fn(retrievedData, relevantData);
  }
  return total / data.relevantDataByQuery.length;
}
