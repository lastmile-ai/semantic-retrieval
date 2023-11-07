import { DataSource } from "../../ingestion/data-sources/dataSource";
import { Document, RawDocument } from "../document";
import { ResourceAccessPolicy } from "../../access-control/resourceAccessPolicy";

export interface DocumentMetadata {
  documentId: string;
  rawDocument?: RawDocument;
  document?: Document;

  // The ids of any root documents from the data sources from which this
  // document was derived (after any number of transformations).
  sourceDocumentIds?: string[];

  collectionId?: string;

  uri: string;
  dataSource?: DataSource;
  name?: string;
  mimeType?: string;
  // The hash of the document content.
  hash?: string;

  // Access policies associated with the document.
  accessPolicies?: ResourceAccessPolicy[];

  // Any JSON-serializable metadata associated with the document.
  metadata?: { [key: string]: string };
  // A general property bag associated with this object.
  attributes?: { [key: string]: string };
}
