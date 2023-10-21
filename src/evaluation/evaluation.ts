import { Document } from "../document/document";

export type RetrievalMetricWithGT = (
  retrievedIds: string[],
  relevantIds: string[]
) => number;

export function evaluateDocumentListRetrievalWithGT(
  retrievedDocuments: Document[],
  relevantDocumentIds: string[],
  metric: RetrievalMetricWithGT
): number {
  const retrievedIds = retrievedDocuments.map(
    (document) => document.documentId
  );
  return metric(retrievedIds, relevantDocumentIds);
}

export const calculateRecall: RetrievalMetricWithGT = (
  retrievedIds: string[],
  relevantIds: string[]
) => {
  const intersection = retrievedIds.filter((id) => relevantIds.includes(id));
  return intersection.length / relevantIds.length;
};

export const calculatePrecision: RetrievalMetricWithGT = (
  retrievedIds: string[],
  relevantIds: string[]
) => {
  const intersection = retrievedIds.filter((id) => relevantIds.includes(id));
  return intersection.length / retrievedIds.length;
};
