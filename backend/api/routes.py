from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.rag_service import RAGService


router = APIRouter()

rag_service = RAGService()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@router.get("/status")
def status():
    """
    Check the status of the ResearchMind service.
    """
    return {
        "status": "running",
        "service": "ResearchMind"
    }


@router.post("/ask")
def ask(request: AskRequest):
    """
    Ask a research question using the RAG pipeline.
    """

    result = rag_service.ask(
        question=request.question,
        top_k=request.top_k
    )

    return result
