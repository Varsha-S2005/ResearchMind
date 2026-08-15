import sys

sys.stdout.reconfigure(
    encoding="utf-8"
)

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

# Number of candidates retrieved before reranking
RETRIEVAL_K = 40


# ============================================================
# RELEVANCE
# ============================================================

def is_relevant(result, evaluation_item):

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

    return (
        document_id in relevant_documents
        and chunk_id in relevant_chunks
    )


# ============================================================
# GET RELEVANT RANKS
# ============================================================

def get_relevant_ranks(
    results,
    evaluation_item
):

    ranks = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_relevant(
            result,
            evaluation_item
        ):

            ranks.append(rank)

    return ranks


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results, evaluation_item):

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

    relevance = [
        is_relevant(
            result,
            evaluation_item
        )
        for result in results
    ]

    def recall_at_k(k):

        retrieved_relevant = sum(
            relevance[:k]
        )

        return min(
            retrieved_relevant / relevant_count,
            1.0
        )

    def precision_at_k(k):

        retrieved_relevant = sum(
            relevance[:k]
        )

        return retrieved_relevant / k

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
# CANDIDATE RECALL
# ============================================================

def calculate_candidate_recall(
    results,
    evaluation_item
):

    relevant_chunks = set(
        evaluation_item.get(
            "relevant_chunks",
            []
        )
    )

    if not relevant_chunks:

        return 0.0

    retrieved_chunks = set()

    for result in results:

        chunk = result["chunk"]

        if chunk.get("document_id") in evaluation_item.get(
            "relevant_documents",
            []
        ):

            retrieved_chunks.add(
                chunk.get("chunk_id")
            )

    found = len(
        relevant_chunks.intersection(
            retrieved_chunks
        )
    )

    return min(
        found / len(relevant_chunks),
        1.0
    )


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

    return {
        metric: sum(
            item[metric]
            for item in all_metrics
        ) / len(all_metrics)
        for metric in metric_names
    }


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
# BASIC RANKING DIAGNOSTIC
# ============================================================

def print_ranking_diagnostic(
    hybrid_top5,
    reranked_results,
    evaluation_item
):

    hybrid_ranks = get_relevant_ranks(
        hybrid_top5,
        evaluation_item
    )

    reranked_ranks = get_relevant_ranks(
        reranked_results,
        evaluation_item
    )

    print()
    print("RANKING DIAGNOSTIC")
    print("-" * 30)

    print(
        f"Hybrid relevant ranks   : "
        f"{hybrid_ranks}"
    )

    print(
        f"Reranked relevant ranks : "
        f"{reranked_ranks}"
    )


# ============================================================
# DETAILED RERANKING DIAGNOSTIC
# ============================================================

def print_reranking_diagnostic(
    question,
    relevant_chunks,
    hybrid_results,
    reranked_results
):
    print()
    print("=" * 70)
    print("DETAILED RERANKING DIAGNOSTIC")
    print("=" * 70)

    print()
    print(f"Question: {question}")
    print()
    print(
        f"Relevant chunks: "
        f"{relevant_chunks}"
    )

    # --------------------------------------------------------
    # HYBRID CANDIDATES
    # --------------------------------------------------------

    print()
    print("HYBRID CANDIDATES")
    print("-" * 70)

    for rank, result in enumerate(
        hybrid_results,
        start=1
    ):
        chunk = result.get(
            "chunk",
            {}
        )

        chunk_id = chunk.get(
            "chunk_id",
            chunk.get("id", "?")
        )

        page = chunk.get(
            "page",
            None
        )

        score = float(
            result.get(
                "score",
                0.0
            )
        )

        marker = (
            " <-- RELEVANT"
            if chunk_id in relevant_chunks
            else ""
        )

        print(
            f"{rank:2}. "
            f"chunk={str(chunk_id):>3} "
            f"page={str(page):<3} "
            f"score={score:.6f}"
            f"{marker}"
        )

    # --------------------------------------------------------
    # RERANKED TOP-5
    # --------------------------------------------------------

    print()
    print("RERANKED TOP-5")
    print("-" * 70)

    for rank, result in enumerate(
        reranked_results,
        start=1
    ):
        chunk = result.get(
            "chunk",
            {}
        )

        chunk_id = chunk.get(
            "chunk_id",
            chunk.get("id", "?")
        )

        page = chunk.get(
            "page",
            None
        )

        cross_score = float(
            result.get(
                "rerank_score",
                0.0
            )
        )

        final_score = float(
            result.get(
                "score",
                0.0
            )
        )

        hybrid_score = float(
            result.get(
                "retrieval_score",
                0.0
            )
        )

        marker = (
            " <-- RELEVANT"
            if chunk_id in relevant_chunks
            else ""
        )

        print(
            f"{rank:2}. "
            f"chunk={str(chunk_id):>3} "
            f"page={str(page):<3} "
            f"cross={cross_score:.6f} "
            f"hybrid={hybrid_score:.6f} "
            f"final={final_score:.6f}"
            f"{marker}"
        )

        text_preview = chunk.get(
            "text",
            ""
        ).replace(
            "\n",
            " "
        )[:180]

        print(
            f"    Text: {text_preview}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ResearchMind Retrieval + Reranking Evaluation"
    )
    print("=" * 70)

    evaluation_dataset = get_evaluation_dataset()

    print(
        f"Evaluation questions: "
        f"{len(evaluation_dataset)}"
    )

    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------

    pdf_path = Path(
        PDF_PATH
    )

    if not pdf_path.exists():

        print(
            f"ERROR: PDF not found: "
            f"{PDF_PATH}"
        )

        return

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
    # MODELS
    # --------------------------------------------------------

    embedder = Embedder()

    vector_store = ChromaStore(
        persist_directory="data/chroma"
    )

    print()
    print(
        "Generating embeddings..."
    )

    embeddings = [
        embedder.embed_text(
            chunk["text"]
        )
        for chunk in chunks
    ]

    vector_store.add_chunks(
        chunks,
        embeddings
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
    # METRICS STORAGE
    # --------------------------------------------------------

    hybrid_metrics_all = []

    reranked_metrics_all = []

    candidate_recalls = []

    # ========================================================
    # QUESTIONS
    # ========================================================

    for question_number, evaluation_item in enumerate(
        evaluation_dataset,
        start=1
    ):

        question = evaluation_item[
            "question"
        ]

        print()
        print("=" * 70)

        print(
            f"Question {question_number}: "
            f"{question}"
        )

        # ====================================================
        # RETRIEVE 40 CANDIDATES
        # ====================================================

        candidate_results = hybrid_retriever.search(
            question,
            top_k=RETRIEVAL_K,
            retrieval_k=RETRIEVAL_K
        )

        # ====================================================
        # CANDIDATE RECALL
        # ====================================================

        candidate_recall = calculate_candidate_recall(
            candidate_results,
            evaluation_item
        )

        candidate_recalls.append(
            candidate_recall
        )

        # ====================================================
        # HYBRID TOP-5
        # ====================================================

        hybrid_top5 = candidate_results[
            :TOP_K
        ]

        hybrid_metrics = calculate_metrics(
            hybrid_top5,
            evaluation_item
        )

        hybrid_metrics_all.append(
            hybrid_metrics
        )

        # ====================================================
        # RERANK 40 -> 5
        # ====================================================

        reranked_results = reranker.rerank(
            question,
            candidate_results,
            top_k=TOP_K
        )

        reranked_metrics = calculate_metrics(
            reranked_results,
            evaluation_item
        )

        reranked_metrics_all.append(
            reranked_metrics
        )

        # ====================================================
        # BASIC DIAGNOSTIC
        # ====================================================

        print_ranking_diagnostic(
            hybrid_top5,
            reranked_results,
            evaluation_item
        )

        # ====================================================
        # DETAILED DIAGNOSTIC
        # ONLY FOR QUESTION 3
        # ====================================================

        if question_number == 3:

            print_reranking_diagnostic(
                question,
                evaluation_item,
                candidate_results,
                reranked_results
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        print()

        print(
            f"Candidate Recall@"
            f"{RETRIEVAL_K}: "
            f"{candidate_recall:.4f}"
        )

        print()

        print(
            "HYBRID TOP-5"
        )

        print(
            "-" * 30
        )

        print_metrics(
            hybrid_metrics
        )

        print()

        print(
            "RERANKED TOP-5"
        )

        print(
            "-" * 30
        )

        print_metrics(
            reranked_metrics
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    hybrid_final = average_metrics(
        hybrid_metrics_all
    )

    reranked_final = average_metrics(
        reranked_metrics_all
    )

    average_candidate_recall = (
        sum(candidate_recalls)
        /
        len(candidate_recalls)
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "FINAL RETRIEVAL RESULTS"
    )

    print("=" * 70)

    print()

    print(
        "HYBRID TOP-5"
    )

    print(
        "-" * 30
    )

    print_metrics(
        hybrid_final
    )

    print()

    print(
        f"HYBRID CANDIDATE RECALL@"
        f"{RETRIEVAL_K}"
    )

    print(
        "-" * 30
    )

    print(
        f"Candidate Recall: "
        f"{average_candidate_recall:.4f}"
    )

    print()

    print(
        "AFTER CROSS-ENCODER RERANKING"
    )

    print(
        "-" * 30
    )

    print_metrics(
        reranked_final
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "RERANKING IMPROVEMENT"
    )

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

        before = hybrid_final[
            metric
        ]

        after = reranked_final[
            metric
        ]

        improvement = (
            after - before
        )

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

    print(
        "INTERPRETATION"
    )

    print("=" * 70)

    if average_candidate_recall >= 0.80:

        print(
            f"Candidate retrieval reached "
            f"the 80% target: "
            f"{average_candidate_recall:.2%}"
        )

        print(
            "Next focus: cross-encoder reranking."
        )

    else:

        print(
            f"Candidate retrieval is below "
            f"the 80% target: "
            f"{average_candidate_recall:.2%}"
        )

        print(
            "Next focus: improve hybrid retrieval."
        )

    if (
        reranked_final["mrr"]
        >
        hybrid_final["mrr"]
    ):

        print(
            "Cross-encoder improved ranking quality."
        )

    elif (
        reranked_final["mrr"]
        <
        hybrid_final["mrr"]
    ):

        print(
            "Cross-encoder reduced ranking quality."
        )

    else:

        print(
            "Cross-encoder did not change MRR."
        )

    print()

    print("=" * 70)

    print(
        "Evaluation completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
