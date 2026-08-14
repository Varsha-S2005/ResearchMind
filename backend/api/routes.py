from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from backend.services.rag_service import RAGService


router = APIRouter()

rag_service = RAGService()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@router.get("/status")
def status():
    """
    Check the status of the ResearchMind service.
    """

    return {
        "status": "ok",
        "service": "ResearchMind"
    }


@router.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Answer a research question using the RAG pipeline.
    """

    return rag_service.ask(
        question=request.question,
        top_k=request.top_k
    )


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Upload and index a research PDF.
    """

    # Check file type
    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF files are supported"
        }

    # Save uploaded PDF
    upload_path = f"data/papers/{file.filename}"

    contents = await file.read()

    with open(upload_path, "wb") as f:
        f.write(contents)

    # Process and index the PDF
    chunks_count = rag_service.ingest_pdf(
        upload_path
    )

    return {
        "message": "PDF uploaded and indexed successfully",
        "filename": file.filename,
        "chunks": chunks_count
    }
