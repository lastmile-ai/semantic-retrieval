import { VectorDB } from "../data-store/vector-DBs/vectorDB";
import { Document } from "../document/document";
import { BaseVectorDBRetriever, VectorDBRetrieverParams } from "./vectorDBRetriever";

/**
 * Base class for retrieving Documents from an underlying VectorDB
 */
export class VectorDBDocumentRetriever<V extends VectorDB> extends BaseVectorDBRetriever<V, Document[]> {
    constructor(vectorDB: V) {
        super(vectorDB);
    }

    protected async _processDocuments(documents: Document[]): Promise<Document[]> {
        return documents
    }
}