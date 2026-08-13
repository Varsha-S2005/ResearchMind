from backend.pipeline.ingestion_pipeline import IngestionPipeline


def test_ingestion_pipeline():
    pipeline = IngestionPipeline()

    pipeline.ingest_pdf("data/papers/sample.pdf")


if __name__ == "__main__":
    test_ingestion_pipeline()
