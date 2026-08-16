from backend.services.rag_service import RAGService
from tests.evaluation.evaluation_dataset import get_evaluation_dataset


def calculate_recall(results, relevant_chunks, k):
    retrieved = {
        result["chunk"]["chunk_id"]
        for result in results[:k]
    }

    relevant = set(relevant_chunks)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def calculate_mrr(results, relevant_chunks):
    relevant = set(relevant_chunks)

    for rank, result in enumerate(results, start=1):
        if result["chunk"]["chunk_id"] in relevant:
            return 1.0 / rank

    return 0.0


def main():

    print("=" * 70)
    print("ResearchMind Cross-Encoder Reranking Evaluation")
    print("=" * 70)

    dataset = get_evaluation_dataset()
    service = RAGService()

    k_values = [5, 10, 20, 40]

    recall_totals = {
        k: 0.0
        for k in k_values
    }

    mrr_total = 0.0

    for number, item in enumerate(dataset, start=1):

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        hybrid_results = service.retriever.search(
            question,
            top_k=40
        )

        reranked_results = service.reranker.rerank(
            question,
            hybrid_results,
            top_k=40
        )

        print()
        print("-" * 70)
        print(f"Question {number}: {question}")

        for k in k_values:

            recall = calculate_recall(
                reranked_results,
                relevant_chunks,
                k
            )

            recall_totals[k] += recall

            print(
                f"Recall@{k:<2}: {recall:.2%}"
            )

        mrr = calculate_mrr(
            reranked_results,
            relevant_chunks
        )

        mrr_total += mrr

        print(f"MRR: {mrr:.4f}")

    count = len(dataset)

    print()
    print("=" * 70)
    print("FINAL RERANKING RESULTS")
    print("=" * 70)

    for k in k_values:

        print(
            f"Recall@{k:<2}: "
            f"{recall_totals[k] / count:.2%}"
        )

    print(
        f"MRR       : "
        f"{mrr_total / count:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
