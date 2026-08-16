from backend.services.rag_service import RAGService
from tests.evaluation.evaluation_dataset import (
    get_evaluation_dataset
)


def calculate_recall(
    retrieved_results: list[dict],
    relevant_chunks: list[int],
    k: int
) -> float:

    retrieved_chunks = {
        result["chunk"]["chunk_id"]
        for result in retrieved_results[:k]
    }

    relevant_chunks_set = set(
        relevant_chunks
    )

    if not relevant_chunks_set:
        return 0.0

    return len(
        retrieved_chunks & relevant_chunks_set
    ) / len(relevant_chunks_set)


def calculate_reciprocal_rank(
    retrieved_results: list[dict],
    relevant_chunks: list[int]
) -> float:

    relevant_chunks_set = set(
        relevant_chunks
    )

    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):

        chunk_id = result[
            "chunk"
        ]["chunk_id"]

        if chunk_id in relevant_chunks_set:
            return 1.0 / rank

    return 0.0


def main():

    print("=" * 70)
    print("ResearchMind Retrieval Recall Evaluation")
    print("=" * 70)

    dataset = get_evaluation_dataset()

    print(
        f"Evaluation questions: {len(dataset)}"
    )

    print()
    print("Initializing RAG service...")

    service = RAGService()

    print("RAG service initialized.")

    k_values = [5, 10, 20, 40]

    recall_totals = {
        k: 0.0
        for k in k_values
    }

    reciprocal_rank_total = 0.0

    for question_number, item in enumerate(
        dataset,
        start=1
    ):

        question = item["question"]

        relevant_chunks = item[
            "relevant_chunks"
        ]

        print()
        print("-" * 70)
        print(
            f"Question {question_number}: "
            f"{question}"
        )

        try:

            results = service.retriever.search(
                question,
                top_k=40
            )

            for k in k_values:

                recall = calculate_recall(
                    results,
                    relevant_chunks,
                    k
                )

                recall_totals[k] += recall

                print(
                    f"Recall@{k:<2}: "
                    f"{recall:.2%}"
                )

            reciprocal_rank = (
                calculate_reciprocal_rank(
                    results,
                    relevant_chunks
                )
            )

            reciprocal_rank_total += (
                reciprocal_rank
            )

            print(
                f"Reciprocal Rank: "
                f"{reciprocal_rank:.4f}"
            )

        except Exception as error:

            print()
            print("RETRIEVAL ERROR")
            print("-" * 70)
            print(error)

    question_count = len(dataset)

    print()
    print("=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    for k in k_values:

        average_recall = (
            recall_totals[k]
            / question_count
        )

        print(
            f"Recall@{k:<2}: "
            f"{average_recall:.2%}"
        )

    mean_reciprocal_rank = (
        reciprocal_rank_total
        / question_count
    )

    print(
        f"MRR       : "
        f"{mean_reciprocal_rank:.4f}"
    )

    print()
    print("=" * 70)
    print("Retrieval evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
