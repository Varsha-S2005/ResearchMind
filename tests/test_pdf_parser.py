from backend.ingestion.pdf_parser import extract_text_from_pdf


pdf_path = "data/papers/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print(f"Number of pages: {len(pages)}")

for page in pages[:3]:
    print("\n--------------------")
    print(f"Page: {page['page_number']}")
    print(page["text"][:500])
    