from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever

from backend.reranking.cross_encoder import CrossEncoderReranker

from tests.evaluation.evaluation_dataset import get_evaluation_dataset


# ============================================================
# Configuration
# ============================================================

PDF_PATH = "data/papers/sample.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_K = 10
TOP_K = 5


# ============================================================
# Relevance Check
# ============================================================

def is_relevant(
    result: dict,
    evaluation_item: dict
) -> bool:
    """
    Determine whether a retrieved chunk is relevant
    according to chunk-level ground truth.
    """

    chunk = result["chunk"]

    document_id = chunk.get("document_id")
    chunk_id = chunk.get("chunk_id")

    relevant_documents = evaluation_item[
        "relevant_documents"
    ]

    relevant_chunks = evaluation_item[
        "relevant_chunks"
    ]

    return (
        document_id in relevant_documents
        and chunk_id in relevant_chunks
    )


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(
    results: list[dict],
    evaluation_item: dict,
    k: int
) -> float:

    relevant_chunks = evaluation_item["relevant_chunks"]

    if not relevant_chunks:
        return 0.0

    retrieved_chunks = set()

    for result in results[:k]:
        chunk_id = result["chunk"].get("chunk_id")

        if chunk_id is not None:
            retrieved_chunks.add(chunk_id)

    relevant_retrieved = (
        retrieved_chunks
        & set(relevant_chunks)
    )

    return len(relevant_retrieved) / len(relevant_chunks)


# ============================================================
# Precision@K
# ============================================================

def precision_at_k(
    results: list[dict],
    evaluation_item: dict,
    k: int
) -> float:

    if k == 0:
        return 0.0

    top_results = results[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        1
        for result in top_results
        if is_relevant(result, evaluation_item)
    )

    return relevant_count / len(top_results)


# ============================================================
# Reciprocal Rank
# ============================================================

def reciprocal_rank(
    results: list[dict],
    evaluation_item: dict
) -> float:

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_relevant(
            result,
            evaluation_item
        ):
            return 1.0 / rank

    return 0.0


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_retrieval():

    print("=" * 70)
    print("ResearchMind Retrieval Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------------

    pages = extract_text_from_pdf(
        PDF_PATH
    )

    print(
        f"\nLoaded {len(pages)} pages."
    )

    # --------------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------------

    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(
        f"Generated {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # 3. Initialize components
    # --------------------------------------------------------

    embedder = Embedder()

    vector_store = ChromaStore(
        persist_directory="data/chroma"
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder
    )

    reranker = CrossEncoderReranker()

    # --------------------------------------------------------
    # 4. Load evaluation questions
    # --------------------------------------------------------

    evaluation_dataset = (
        get_evaluation_dataset()
    )

    print(
        f"Evaluation questions: "
        f"{len(evaluation_dataset)}"
    )

    print("\n" + "-" * 70)

    # --------------------------------------------------------
    # 5. Metric accumulators
    # --------------------------------------------------------

    total_recall_1 = 0.0
    total_recall_3 = 0.0
    total_recall_5 = 0.0

    total_precision_1 = 0.0
    total_precision_3 = 0.0
    total_precision_5 = 0.0

    total_mrr = 0.0

    # --------------------------------------------------------
    # 6. Evaluate each question
    # --------------------------------------------------------

    for question_number, item in enumerate(
        evaluation_dataset,
        start=1
    ):

        question = item["question"]

        print(
            f"\nQuestion {question_number}: "
            f"{question}"
        )

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        retrieved_results = hybrid_retriever.search(
            question,
            top_k=TOP_K,
            retrieval_k=RETRIEVAL_K
        )

        # ----------------------------------------------------
        # Reranking
        # ----------------------------------------------------

        try:

            reranked_results = reranker.rerank(
                question,
                retrieved_results,
                top_k=TOP_K
            )

        except Exception as error:

            print(
                "Warning: Reranking failed."
            )

            print(
                f"Reason: {error}"
            )

            reranked_results = (
                retrieved_results
            )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        r1 = recall_at_k(
            reranked_results,
            item,
            1
        )

        r3 = recall_at_k(
            reranked_results,
            item,
            3
        )

        r5 = recall_at_k(
            reranked_results,
            item,
            5
        )

        p1 = precision_at_k(
            reranked_results,
            item,
            1
        )

        p3 = precision_at_k(
            reranked_results,
            item,
            3
        )

        p5 = precision_at_k(
            reranked_results,
            item,
            5
        )

        rr = reciprocal_rank(
            reranked_results,
            item
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        total_recall_1 += r1
        total_recall_3 += r3
        total_recall_5 += r5

        total_precision_1 += p1
        total_precision_3 += p3
        total_precision_5 += p5

        total_mrr += rr

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"Recall@1  : {r1:.2f}"
        )

        print(
            f"Recall@3  : {r3:.2f}"
        )

        print(
            f"Recall@5  : {r5:.2f}"
        )

        print(
            f"Precision@1 : {p1:.2f}"
        )

        print(
            f"Precision@3 : {p3:.2f}"
        )

        print(
            f"Precision@5 : {p5:.2f}"
        )

        print(
            f"Reciprocal Rank : {rr:.2f}"
        )

        # ----------------------------------------------------
        # Show retrieved chunks
        # ----------------------------------------------------

        print(
            "\nRetrieved chunks:"
        )

        for rank, result in enumerate(
            reranked_results,
            start=1
        ):

            chunk = result["chunk"]

            document_id = chunk.get(
                "document_id",
                "unknown"
            )

            page_number = chunk.get(
                "page_number",
                "unknown"
            )

            chunk_id = chunk.get(
                "chunk_id",
                "unknown"
            )

            relevant = is_relevant(
                result,
                item
            )

            marker = (
                " <-- RELEVANT"
                if relevant
                else ""
            )

            print(
                f"  {rank}. "
                f"Document={document_id} "
                f"Page={page_number} "
                f"Chunk={chunk_id}"
                f"{marker}"
            )

    # ========================================================
    # Final Results
    # ========================================================

    question_count = len(
        evaluation_dataset
    )

    if question_count == 0:
        print(
            "\nNo evaluation questions found."
        )
        return

    avg_recall_1 = (
        total_recall_1 / question_count
    )

    avg_recall_3 = (
        total_recall_3 / question_count
    )

    avg_recall_5 = (
        total_recall_5 / question_count
    )

    avg_precision_1 = (
        total_precision_1 / question_count
    )

    avg_precision_3 = (
        total_precision_3 / question_count
    )

    avg_precision_5 = (
        total_precision_5 / question_count
    )

    mrr = (
        total_mrr / question_count
    )

    # ========================================================
    # Print Final Results
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\nRecall@1      : {avg_recall_1:.4f}"
    )

    print(
        f"Recall@3      : {avg_recall_3:.4f}"
    )

    print(
        f"Recall@5      : {avg_recall_5:.4f}"
    )

    print(
        f"\nPrecision@1   : {avg_precision_1:.4f}"
    )

    print(
        f"Precision@3   : {avg_precision_3:.4f}"
    )

    print(
        f"Precision@5   : {avg_precision_5:.4f}"
    )

    print(
        f"\nMRR           : {mrr:.4f}"
    )

    print("\n")
    print(
        "Evaluation completed."
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    evaluate_retrieval()
