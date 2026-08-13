import fitz


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.
    """

    document = fitz.open(pdf_path) #opens the pdf file 

    pages = []

    for page_number, page in enumerate(document, start=1): #goes through all pages
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    document.close()

    return pages