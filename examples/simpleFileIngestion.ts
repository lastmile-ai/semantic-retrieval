// TODO: These imports should be from actual lastmile retrieval package
import { AccessPassport } from '../src/access-control/accessPassport';
import { AlwaysAllowDocumentAccessPolicyFactory } from '../src/access-control/alwaysAllowDocumentAccessPolicyFactory';
import { PineconeVectorDB } from '../src/data-store/vector-DBs.ts/pineconeVectorDB';
import { VectorDBTextQuery } from '../src/data-store/vector-DBs.ts/vectorDB';
import { InMemoryDocumentMetadataDB } from '../src/document/metadata/InMemoryDocumentMetadataDB';
import { FileSystem } from '../src/ingestion/data-sources/dataSource';
import { SimpleDocumentParser } from '../src/ingestion/document-parsers/simpleDocumentParser';

async function createIndex() {
    const fileSystem = new FileSystem('./example_docs');
    const rawDocuments = await fileSystem.loadDocuments();
    const metadataDB = new InMemoryDocumentMetadataDB();
    const parsedDocuments = await SimpleDocumentParser.parseDocuments(rawDocuments, {
        metadataDB,
        accessControlPolicyFactory: new AlwaysAllowDocumentAccessPolicyFactory(),
    });

    return await PineconeVectorDB.fromDocuments(parsedDocuments, metadataDB);
}

async function main() {
    const vectorDB = await createIndex();
    const query: VectorDBTextQuery = {text: "How do I use parameters in a workbook?"};
    const accessPassport = new AccessPassport();
    const res = await vectorDB.query(accessPassport, query);
    console.log(res);
}

main();

export {};
