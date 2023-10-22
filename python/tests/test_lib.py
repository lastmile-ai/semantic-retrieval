from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem


def test_create_index():
    file_system = FileSystem("./example_docs")
    raw_documents = file_system.load_documents()

    # TODO: Continue making stubs and essentially getting the demo as a test case (similar to localFileIngestion.ts right now)
    # Then can start to write the actual implementation / split the work

    assert True
