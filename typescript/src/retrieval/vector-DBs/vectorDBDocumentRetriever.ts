import { Document } from "../../document/document";
import { RetrieverProcessDocumentsEvent } from "../../utils/callbacks";
import {
  BaseVectorDBRetriever,
  VectorDBRetrieverParams,
} from "./vectorDBRetriever";

/**
 * Base class for retrieving Documents from an underlying VectorDB
 */
export class VectorDBDocumentRetriever extends BaseVectorDBRetriever<
  Document[]
> {
  constructor(params: VectorDBRetrieverParams) {
    super(params);
  }

  protected async processDocuments(documents: Document[]): Promise<Document[]> {
    const event: RetrieverProcessDocumentsEvent = {
      name: "onRetrieverProcessDocuments",
      documents,
    };
    await this.callbackManager?.runCallbacks(event);

    return documents;
  }
}
