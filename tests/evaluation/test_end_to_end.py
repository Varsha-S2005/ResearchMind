from backend.services.rag_service import RAGService


QUESTIONS = [
    "What are the main challenges of federated learning in vehicular networks?",
    "Why is non-IID data a problem in vehicular federated learning?",
    "What security threats affect federated learning in vehicular networks?",
    "What are the limitations of centralized federated learning in vehicular networks?",
    "What challenges are associated with Byzantine participants in vehicular federated learning?",
    "What regulatory issues affect federated learning in automotive cybersecurity?",
    "What communication problems affect federated learning convergence?",
    "What are the evaluation pitfalls in current vehicular federated learning research?",
]


def main():

    print("=" * 70)
    print("ResearchMind END-TO-END RAG TEST")
    print("=" * 70)

    service = RAGService()

    for number, question in enumerate(QUESTIONS, start=1):

        print()
        print("-" * 70)
        print(f"QUESTION {number}")
        print(question)
        print("-" * 70)

        try:

            result = service.ask(
                question,
                top_k=5
            )

            print()
            print("ANSWER")
            print("-" * 70)
            print(result["answer"])

            print()
            print("SOURCES")
            print("-" * 70)

            for source in result["sources"]:

                print(
                    f"{source['source_id']} | "
                    f"Document: {source['document_id']} | "
                    f"Page: {source['page_number']} | "
                    f"Chunk: {source['chunk_id']}"
                )

        except Exception as error:

            print()
            print("RAG ERROR")
            print("-" * 70)
            print(error)

    print()
    print("=" * 70)
    print("END-TO-END TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
