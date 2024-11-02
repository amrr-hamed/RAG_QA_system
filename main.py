import os
from dotenv import load_dotenv
from modules.pdf_handler import PDFHandler
from modules.vector_store import VectorStore
from modules.qa_chain import QAChain

def main():
    # Load environment variables
    load_dotenv()

    # Initialize components
    pdf_handler = PDFHandler()
    vector_store = VectorStore(
        persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
        collection_name="documents"
    )
    qa_chain = QAChain()

    # Example function to process a PDF file
    def process_pdf(pdf_path: str):
        """Process a PDF file and add it to the vector store."""
        # Extract text from PDF
        text = pdf_handler.extract_text_from_pdf(pdf_path)
        
        # Split text into chunks
        chunks = pdf_handler.split_text(text)
        
        # Add chunks to vector store
        vector_store.add_documents(chunks)

    # Example function to ask a question
    def ask_question(question: str):
        """Ask a question and get an answer."""
        try:
            # Get retriever from vector store
            retriever = vector_store.as_retriever()
            
            # Process query
            answer, sources = qa_chain.process_query(retriever, question)
            
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")
            print("Sources:")
            for i, doc in enumerate(sources):
                print(f"Source {i+1}: {doc.page_content[:200]}...\n")
                
        except Exception as e:
            print(f"Error: {str(e)}")

    # Run example usage if the script is executed directly
    if __name__ == "__main__":
        # Prompt the user to input the PDF path and question
        pdf_path = input("Enter the path to the PDF file: ").strip()
        process_pdf(pdf_path)
        
        # Ask a question
        question = input("Enter your question: ").strip()
        ask_question(question)

# Entry point
main()
