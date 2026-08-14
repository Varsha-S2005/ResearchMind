from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

from backend.services.rag_service import RAGService


router = APIRouter()

rag_service = RAGService()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@router.get("/status")
def status():
    return {
        "status": "running",
        "service": "ResearchMind"
    }


@router.post("/ask")
def ask(request: AskRequest):

    result = rag_service.ask(
        question=request.question,
        top_k=request.top_k
    )

    return result


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a research paper and add it to the RAG system.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    os.makedirs(
        "data/uploads",
        exist_ok=True
    )

    file_path = os.path.join(
        "data/uploads",
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    chunk_count = rag_service.ingest_pdf(
        file_path,
	file.filename
    )

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "chunks_added": chunk_count
    }
