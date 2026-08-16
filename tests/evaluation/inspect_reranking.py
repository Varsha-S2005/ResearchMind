from backend.services.rag_service import RAGService
from tests.evaluation.evaluation_dataset import get_evaluation_dataset


def main():

    print("=" * 70)
    print("CROSS-ENCODER RERANKING DIAGNOSTIC")
    print("=" * 70)

    dataset = get_evaluation_dataset()
    service = RAGService()

    for number, item in enumerate(dataset, start=1):

        question = item["question"]
        relevant_chunks = set(item["relevant_chunks"])

        print()
        print("-" * 70)
        print(f"QUESTION {number}")
        print(question)
        print(f"Relevant chunks: {sorted(relevant_chunks)}")

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
        print("RELEVANT CHUNK RANK MOVEMENT")
        print("-" * 70)

        found = False

        for final_rank, result in enumerate(
            reranked_results,
            start=1
        ):

            chunk_id = result["chunk"]["chunk_id"]

            if chunk_id in relevant_chunks:

                found = True

                print(
                    f"chunk={chunk_id:3} "
                    f"hybrid_rank={result['hybrid_rank']:3} "
                    f"cross_rank={result['cross_encoder_rank']:3} "
                    f"final_rank={final_rank:3} "
                    f"cross_score={result['rerank_score']:.4f}"
                )

        if not found:
            print("No relevant chunks found.")

    print()
    print("=" * 70)
    print("RERANKING DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
