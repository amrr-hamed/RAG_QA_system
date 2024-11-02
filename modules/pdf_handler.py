import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFHandler:
    def __init__(self, chunk_size=512, chunk_overlap=10):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                # Extract raw text
                page_text = page.get_text("text")
                # Replace multiple spaces with single space
                page_text = ' '.join(page_text.split())
                # Append cleaned text to full text
                text += page_text + " "
            
            pdf_document.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def split_text(self, text: str) -> list:
        """Split text into chunks."""
        return self.text_splitter.split_text(text)
