from pathlib import Path
import fitz


def extract_text_from_pdf(
    pdf_path: str,
    document_id: str | None = None
) -> list[dict]:
    """
    Extract text from a PDF while preserving:
    - document identity
    - page numbers
    - page text

    If document_id is not provided, the PDF filename
    without its extension is used.
    """

    if document_id is None:
        document_id = Path(pdf_path).stem

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append({
            "document_id": document_id,
            "page_number": page_number,
            "text": text.strip()
        })

    document.close()

    return pages
