"""
Ground-truth evaluation dataset for ResearchMind.

Ground truth is defined at chunk level rather than page level.
A retrieved chunk is considered relevant when its chunk_id
appears in the relevant_chunks list.
"""

EVALUATION_DATASET = [

    {
        "question": (
            "What are the main challenges of federated learning "
            "in vehicular networks?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            1, 2, 8, 14, 16, 17
        ]
    },

    {
        "question": (
            "Why is non-IID data a problem in vehicular "
            "federated learning?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            16
        ]
    },

    {
        "question": (
            "What security threats affect federated learning "
            "in vehicular networks?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            2, 8, 16
        ]
    },

    {
        "question": (
            "What are the limitations of centralized federated "
            "learning in vehicular networks?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            8
        ]
    },

    {
        "question": (
            "What are the challenges of decentralized or "
            "gossip-based federated learning?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            8
        ]
    },

    {
        "question": (
            "Why are real-time CAN bus constraints important "
            "for vehicular federated learning?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            14, 16
        ]
    },

    {
        "question": (
            "What communication problems affect federated "
            "learning convergence?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            14
        ]
    },

    {
        "question": (
            "What are the evaluation pitfalls in current "
            "vehicular federated learning research?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            14, 16
        ]
    },

    {
        "question": (
            "What challenges are associated with Byzantine "
            "participants in vehicular federated learning?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            16
        ]
    },

    {
        "question": (
            "What regulatory issues affect federated learning "
            "in automotive cybersecurity?"
        ),
        "relevant_documents": ["sample"],
        "relevant_chunks": [
            17
        ]
    }
]


def get_evaluation_dataset():
    """
    Return the complete evaluation dataset.
    """
    return EVALUATION_DATASET
