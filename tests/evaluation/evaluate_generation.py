from backend.services.rag_service import RAGService
from tests.evaluation.evaluation_dataset import (
    get_evaluation_dataset
)


def validate_citations(
    answer: str,
    sources: list[dict]
) -> tuple[bool, list[str]]:

    valid_source_ids = {
        source["source_id"]
        for source in sources
    }

    cited_sources = []

    for source_number in range(
        1,
        len(sources) + 1
    ):
        citation = f"[Source {source_number}]"

        if citation in answer:
            cited_sources.append(citation)

    invalid_citations = []

    # Find citation-like references in the answer.
    words = answer.replace(
        "[",
        " ["
    ).split()

    for word in words:

        if word.startswith("[Source") and word.endswith("]"):

            if word not in valid_source_ids:
                invalid_citations.append(word)

    return (
        len(invalid_citations) == 0,
        cited_sources
    )


def main():

    print("=" * 70)
    print("ResearchMind Generation Evaluation")
    print("=" * 70)

    evaluation_dataset = get_evaluation_dataset()

    print(
        f"Evaluation questions: "
        f"{len(evaluation_dataset)}"
    )

    print()
    print("Initializing RAG service...")

    service = RAGService()

    print("RAG service initialized.")

    total_questions = len(
        evaluation_dataset
    )

    valid_answers = 0
    cited_answers = 0
    valid_citations = 0

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

        try:

            result = service.ask(
                question,
                top_k=5
            )

            answer = result.get(
                "answer",
                ""
            )

            sources = result.get(
                "sources",
                []
            )

            # ---------------------------------------------
            # Answer validation
            # ---------------------------------------------

            answer_valid = (
                isinstance(answer, str)
                and len(answer.strip()) > 0
            )

            if answer_valid:
                valid_answers += 1

            # ---------------------------------------------
            # Citation validation
            # ---------------------------------------------

            citations_valid, cited_sources = (
                validate_citations(
                    answer,
                    sources
                )
            )

            if cited_sources:
                cited_answers += 1

            if citations_valid:
                valid_citations += 1

            # ---------------------------------------------
            # Output
            # ---------------------------------------------

            print()
            print("ANSWER")
            print("-" * 70)
            print(answer)

            print()
            print("SOURCES")
            print("-" * 70)

            for source in sources:

                print(
                    f"{source['source_id']}: "
                    f"document={source['document_id']} "
                    f"page={source['page_number']} "
                    f"chunk={source['chunk_id']} "
                    f"score={source['score']:.6f}"
                )

            print()
            print("GENERATION DIAGNOSTIC")
            print("-" * 70)

            print(
                f"Non-empty answer : "
                f"{answer_valid}"
            )

            print(
                f"Citations found  : "
                f"{cited_sources}"
            )

            print(
                f"Citations valid  : "
                f"{citations_valid}"
            )

        except Exception as error:

            print()
            print("GENERATION ERROR")
            print("-" * 70)
            print(error)

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL GENERATION RESULTS")
    print("=" * 70)

    print(
        f"Non-empty answers : "
        f"{valid_answers}/{total_questions} "
        f"({valid_answers / total_questions:.2%})"
    )

    print(
        f"Answers with citations : "
        f"{cited_answers}/{total_questions} "
        f"({cited_answers / total_questions:.2%})"
    )

    print(
        f"Valid citations : "
        f"{valid_citations}/{total_questions} "
        f"({valid_citations / total_questions:.2%})"
    )

    print()
    print("=" * 70)
    print("Generation evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
