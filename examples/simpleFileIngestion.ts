import { AlwaysAllowDocumentAccessPolicyFactory } from '../src/access-control/alwaysAllowDocumentAccessPolicyFactory';
import { InMemoryDocumentMetadataDB } from '../src/document/metadata/InMemoryDocumentMetadataDB';
import { FileSystem } from '../src/ingestion/data-sources/dataSource';

async function main() {
    const fileSystem = new FileSystem('./docs');
    const rawDocuments = await fileSystem.loadDocuments();
    const parsedDocuments = await SimpleDocumentParser.parseDocuments(rawDocuments, {
        metadataDB: new InMemoryDocumentMetadataDB(),
        accessControlPolicyFactory: new AlwaysAllowDocumentAccessPolicyFactory(),
    });
}

main();

export {};
