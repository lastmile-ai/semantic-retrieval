import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { BaseRetriever } from "./retriever";

export abstract class BaseDocumentRetriever<Q> extends BaseRetriever<
  Document[],
  Q
> {
  constructor(metadataDB: DocumentMetadataDB) {
    super(metadataDB);
  }

  /**
   * Perform any post-processing on the retrieved Documents.
   * @param documents The array of retrieved Documents to post-process.
   * @returns A promise that resolves to post-processed data.
   */
  protected async processDocuments(documents: Document[]): Promise<Document[]> {
    return documents;
  }
}
