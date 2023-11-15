import { DocumentMetadata } from "./documentMetadata";

export type DocumentMetadataQuery = {
  type: "metadata";
  metadataKey: string;
  metadataValue: string;
  matchType: "exact" | "includes";
};

export type DocumentStringFieldQuery = {
  type: "string_field";
  fieldName: keyof Omit<
    DocumentMetadata,
    | "document"
    | "rawDocument"
    | "dataSource"
    | "accessPolicies"
    | "attributes"
    | "metadata"
  >;
  fieldValue: string;
  matchType: "exact" | "includes";
};

export type DocumentMetadataDBQuery =
  | DocumentMetadataQuery
  | DocumentStringFieldQuery;

export interface DocumentMetadataDB {
  // TODO: saqadri - implement a Postgres implementation of this interface.
  getMetadata(documentId: string): Promise<DocumentMetadata>;
  setMetadata(documentId: string, metadata: DocumentMetadata): Promise<void>;

  queryDocumentIds(query: DocumentMetadataDBQuery): Promise<string[]>;
}
