from fastapi import FastAPI

app = FastAPI(
    title="ResearchMind API",
    description="RAG system for scientific research papers",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "ResearchMind API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }