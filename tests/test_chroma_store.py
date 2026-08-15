import shutil
from pathlib import Path

from backend.vectorstore.chroma_store import ChromaStore


def test_chroma_store():

    test_directory = Path("data/test_chroma")

    # Remove previous test database so every test starts clean.
    if test_directory.exists():
        shutil.rmtree(test_directory)

    store = ChromaStore(
        persist_directory=str(test_directory),
        collection_name="test_collection"
    )

    chunks = [
        {
            "document_id": "test_document",
            "chunk_id": 1,
            "page_number": 1,
            "text": "Federated learning protects data privacy."
        },
        {
            "document_id": "test_document",
            "chunk_id": 2,
            "page_number": 2,
            "text": "Intrusion detection identifies malicious network activity."
        }
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    store.add_chunks(
        chunks,
        embeddings
    )

    assert store.collection.count() == 2

    results = store.collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    assert len(results["documents"]) == 2
    assert len(results["metadatas"]) == 2
