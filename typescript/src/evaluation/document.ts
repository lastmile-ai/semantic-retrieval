import { Document } from "../document/document";

import {
  RetrievalMetric,
  calculateRecall,
  calculatePrecision,
} from "./evaluation";

function getRetrievedFragmentIds(retrievedDocuments: Document[]): Set<string> {
  const retrievedFragmentIds = new Set<string>();
  retrievedDocuments.forEach((document) => {
    document.fragments.forEach((fragment) => {
      retrievedFragmentIds.add(fragment.fragmentId);
    });
  });
  return retrievedFragmentIds;
}

export const calculateRetrievedFragmentRecall: RetrievalMetric<
  Document[],
  string[]
> = {
  fn: (
    retrievedDocuments: Document[],
    relevantFragmentIds: string[]
  ): number => {
    const retrievedFragmentIds = getRetrievedFragmentIds(retrievedDocuments);
    return calculateRecall(retrievedFragmentIds, new Set(relevantFragmentIds));
  },
  name: "documentFragmentRecall",
};

export const calculateRetrievedFragmentPrecision: RetrievalMetric<
  Document[],
  string[]
> = {
  fn: (
    retrievedDocuments: Document[],
    relevantFragmentIds: string[]
  ): number => {
    const retrievedFragmentIds = getRetrievedFragmentIds(retrievedDocuments);
    return calculatePrecision(
      retrievedFragmentIds,
      new Set(relevantFragmentIds)
    );
  },
  name: "documentFragmentPrecision",
};
