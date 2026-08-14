from pathlib import Path

from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever

from backend.reranking.cross_encoder import CrossEncoderReranker

from tests.evaluation.evaluation_dataset import get_evaluation_dataset


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/papers/sample.pdf"

TOP_K = 5
RETRIEVAL_K = 20
# ============================================================
# RELEVANCE CHECK
# ============================================================

def is_relevant(result, evaluation_item):
    """
    A retrieved chunk is relevant when:
    - document_id matches
    - chunk_id appears in relevant_chunks
    """

    chunk = result["chunk"]

    document_id = chunk.get("document_id")
    chunk_id = chunk.get("chunk_id")

    relevant_documents = evaluation_item.get(
        "relevant_documents",
        []
    )

    relevant_chunks = evaluation_item.get(
        "relevant_chunks",
        []
    )

    if document_id not in relevant_documents:
        return False

    if chunk_id not in relevant_chunks:
        return False

    return True


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(results, evaluation_item):
    """
    Calculate:

    Recall@1
    Recall@3
    Recall@5

    Precision@1
    Precision@3
    Precision@5

    MRR
    """

    relevant_chunks = evaluation_item.get(
        "relevant_chunks",
        []
    )

    relevant_count = len(relevant_chunks)

    if relevant_count == 0:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "precision@1": 0.0,
            "precision@3": 0.0,
            "precision@5": 0.0,
            "mrr": 0.0
        }

    relevance = []

    for result in results:

        relevance.append(
            is_relevant(
                result,
                evaluation_item
            )
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    def recall_at_k(k):

        retrieved_relevant = sum(
            relevance[:k]
        )

        return min(
            retrieved_relevant / relevant_count,
            1.0
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    def precision_at_k(k):

        retrieved_relevant = sum(
            relevance[:k]
        )

        return retrieved_relevant / k

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    reciprocal_rank = 0.0

    for rank, relevant in enumerate(
        relevance,
        start=1
    ):

        if relevant:

            reciprocal_rank = 1.0 / rank

            break

    return {
        "recall@1": recall_at_k(1),
        "recall@3": recall_at_k(3),
        "recall@5": recall_at_k(5),

        "precision@1": precision_at_k(1),
        "precision@3": precision_at_k(3),
        "precision@5": precision_at_k(5),

        "mrr": reciprocal_rank
    }


# ============================================================
# AVERAGE METRICS
# ============================================================

def average_metrics(all_metrics):

    if not all_metrics:
        return {}

    metric_names = [
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@1",
        "precision@3",
        "precision@5",
        "mrr"
    ]

    averages = {}

    for metric in metric_names:

        averages[metric] = (
            sum(
                item[metric]
                for item in all_metrics
            )
            / len(all_metrics)
        )

    return averages


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(metrics):

    print(
        f"Recall@1      : "
        f"{metrics['recall@1']:.4f}"
    )

    print(
        f"Recall@3      : "
        f"{metrics['recall@3']:.4f}"
    )

    print(
        f"Recall@5      : "
        f"{metrics['recall@5']:.4f}"
    )

    print()

    print(
        f"Precision@1   : "
        f"{metrics['precision@1']:.4f}"
    )

    print(
        f"Precision@3   : "
        f"{metrics['precision@3']:.4f}"
    )

    print(
        f"Precision@5   : "
        f"{metrics['precision@5']:.4f}"
    )

    print()

    print(
        f"MRR           : "
        f"{metrics['mrr']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ResearchMind Retrieval + Reranking Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation dataset
    # --------------------------------------------------------

    evaluation_dataset = get_evaluation_dataset()

    print(
        f"Evaluation questions: "
        f"{len(evaluation_dataset)}"
    )

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():

        print(
            f"\nERROR: PDF not found: {PDF_PATH}"
        )

        return

    print(
        f"\nLoading document: {PDF_PATH}"
    )

    document_id = pdf_path.stem

    pages = extract_text_from_pdf(
        str(pdf_path),
        document_id=document_id
    )

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    print(
        f"Loaded {len(pages)} pages."
    )

    print(
        f"Generated {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Initialize embedding model
    # --------------------------------------------------------

    embedder = Embedder()

    # --------------------------------------------------------
    # Initialize ChromaDB
    # --------------------------------------------------------

    vector_store = ChromaStore(
        persist_directory="data/chroma"
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print()
    print("Generating embeddings...")

    embeddings = []

    for chunk in chunks:

        embedding = embedder.embed_text(
            chunk["text"]
        )

        embeddings.append(
            embedding
        )

    # --------------------------------------------------------
    # Store chunks
    # --------------------------------------------------------

    vector_store.add_chunks(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Initialize BM25
    # --------------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks
    )

    # --------------------------------------------------------
    # Initialize Hybrid Retriever
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder
    )

    # --------------------------------------------------------
    # Initialize Cross Encoder
    # --------------------------------------------------------

    reranker = CrossEncoderReranker()

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    hybrid_metrics_all = []

    reranked_metrics_all = []

    # ========================================================
    # EVALUATE QUESTIONS
    # ========================================================

    for question_number, evaluation_item in enumerate(
        evaluation_dataset,
        start=1
    ):

        question = evaluation_item["question"]

        print()
        print("=" * 70)

        print(
            f"Question {question_number}: "
            f"{question}"
        )

        print()

        print(
            f"Relevant document(s): "
            f"{evaluation_item.get('relevant_documents', [])}"
        )

        print(
            f"Relevant chunk(s): "
            f"{evaluation_item.get('relevant_chunks', [])}"
        )

        # ====================================================
        # HYBRID RETRIEVAL
        # ====================================================

        hybrid_results = hybrid_retriever.search(
            question,
            top_k=TOP_K,
            retrieval_k=RETRIEVAL_K
        )

        hybrid_result_metrics = calculate_metrics(
            hybrid_results,
            evaluation_item
        )

        hybrid_metrics_all.append(
            hybrid_result_metrics
        )

        # ====================================================
        # CROSS ENCODER RERANKING
        # ====================================================

        reranked_results = reranker.rerank(
            question,
            hybrid_results,
            top_k=TOP_K
        )

        reranked_result_metrics = calculate_metrics(
            reranked_results,
            evaluation_item
        )

        reranked_metrics_all.append(
            reranked_result_metrics
        )

        # ====================================================
        # HYBRID RESULTS
        # ====================================================

        print()
        print("HYBRID RETRIEVAL")
        print("-" * 30)

        print_metrics(
            hybrid_result_metrics
        )

        # ====================================================
        # RERANKED RESULTS
        # ====================================================

        print()
        print("AFTER CROSS-ENCODER RERANKING")
        print("-" * 30)

        print_metrics(
            reranked_result_metrics
        )

        # ====================================================
        # SHOW HYBRID CHUNKS
        # ====================================================

        print()
        print("Retrieved chunks:")

        for rank, result in enumerate(
            hybrid_results,
            start=1
        ):

            chunk = result["chunk"]

            relevant_marker = ""

            if is_relevant(
                result,
                evaluation_item
            ):

                relevant_marker = " <-- RELEVANT"

            print(
                f"  {rank}. "
                f"Document={chunk.get('document_id')} "
                f"Page={chunk.get('page_number')} "
                f"Chunk={chunk.get('chunk_id')} "
                f"Score={result.get('score', 0):.4f}"
                f"{relevant_marker}"
            )

        # ====================================================
        # SHOW RERANKED CHUNKS
        # ====================================================

        print()
        print("Reranked chunks:")

        for rank, result in enumerate(
            reranked_results,
            start=1
        ):

            chunk = result["chunk"]

            relevant_marker = ""

            if is_relevant(
                result,
                evaluation_item
            ):

                relevant_marker = " <-- RELEVANT"

            print(
                f"  {rank}. "
                f"Document={chunk.get('document_id')} "
                f"Page={chunk.get('page_number')} "
                f"Chunk={chunk.get('chunk_id')} "
                f"Score={result.get('score', 0):.4f}"
                f"{relevant_marker}"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    hybrid_final = average_metrics(
        hybrid_metrics_all
    )

    reranked_final = average_metrics(
        reranked_metrics_all
    )

    print()
    print("=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    print()
    print("HYBRID RETRIEVAL")
    print("-" * 30)

    print_metrics(
        hybrid_final
    )

    # --------------------------------------------------------
    # Reranked
    # --------------------------------------------------------

    print()
    print("AFTER CROSS-ENCODER RERANKING")
    print("-" * 30)

    print_metrics(
        reranked_final
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    print()
    print("=" * 70)
    print("RERANKING IMPROVEMENT")
    print("=" * 70)

    metrics_to_compare = [
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@1",
        "precision@3",
        "precision@5",
        "mrr"
    ]

    for metric in metrics_to_compare:

        before = hybrid_final[metric]

        after = reranked_final[metric]

        improvement = after - before

        print(
            f"{metric:<15}: "
            f"{before:.4f} -> "
            f"{after:.4f} "
            f"({improvement:+.4f})"
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if reranked_final["mrr"] > hybrid_final["mrr"]:

        print(
            "Cross-encoder reranking improved "
            "the overall ranking quality."
        )

    elif reranked_final["mrr"] < hybrid_final["mrr"]:

        print(
            "Cross-encoder reranking reduced "
            "the overall ranking quality."
        )

    else:

        print(
            "Cross-encoder reranking did not "
            "change the overall MRR."
        )

    if (
        reranked_final["recall@1"]
        >
        hybrid_final["recall@1"]
    ):

        print(
            "Reranking improved Recall@1."
        )

    elif (
        reranked_final["recall@1"]
        <
        hybrid_final["recall@1"]
    ):

        print(
            "Reranking reduced Recall@1."
        )

    else:

        print(
            "Reranking did not change Recall@1."
        )

    print()
    print("=" * 70)
    print("Evaluation completed.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
