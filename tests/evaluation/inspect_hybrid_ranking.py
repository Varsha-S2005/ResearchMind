from pathlib import Path

from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever

from tests.evaluation.evaluation_dataset import get_evaluation_dataset


PDF_PATH = "data/papers/sample.pdf"
RETRIEVAL_K = 40


def is_relevant(result, evaluation_item):
    chunk = result["chunk"]

    return (
        chunk.get("document_id")
        in evaluation_item.get("relevant_documents", [])
        and
        chunk.get("chunk_id")
        in evaluation_item.get("relevant_chunks", [])
    )


def main():

    print("=" * 70)
    print("HYBRID RETRIEVAL RANKING DIAGNOSTIC")
    print("=" * 70)

    dataset = get_evaluation_dataset()

    pdf_path = Path(PDF_PATH)

    pages = extract_text_from_pdf(
        str(pdf_path),
        document_id=pdf_path.stem
    )

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    print(f"Pages : {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    embedder = Embedder()

    vector_store = ChromaStore(
        persist_directory="data/chroma"
    )

    embeddings = [
        embedder.embed_text(chunk["text"])
        for chunk in chunks
    ]

    vector_store.add_chunks(
        chunks,
        embeddings
    )

    bm25 = BM25Retriever(chunks)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_store=vector_store,
        embedder=embedder
    )

    print()
    print("=" * 70)
    print("INSPECTING TOP-40 CANDIDATE RANKING")
    print("=" * 70)

    for number, item in enumerate(dataset, start=1):

        question = item["question"]

        results = hybrid.search(
            question,
            top_k=40,
            retrieval_k=40
        )

        print()
        print("-" * 70)
        print(f"QUESTION {number}")
        print(question)

        print()
        print(
            "Relevant chunks:",
            item.get("relevant_chunks", [])
        )

        print()
        print("RANKED CANDIDATES")
        print("-" * 70)

        for rank, result in enumerate(
            results,
            start=1
        ):

            chunk = result["chunk"]

            relevant = is_relevant(
                result,
                item
            )

            marker = " <-- RELEVANT" if relevant else ""

            print(
                f"{rank:2}. "
                f"chunk={chunk.get('chunk_id'):>3} "
                f"page={chunk.get('page_number'):>2} "
                f"score={result.get('score', 0):.6f}"
                f"{marker}"
            )

        relevant_ranks = []

        for rank, result in enumerate(
            results,
            start=1
        ):
            if is_relevant(result, item):
                relevant_ranks.append(rank)

        print()

        if relevant_ranks:
            print(
                "Relevant chunk ranks:",
                relevant_ranks
            )
        else:
            print(
                "WARNING: No relevant chunk found "
                "inside Top-40."
            )


if __name__ == "__main__":
    main()
