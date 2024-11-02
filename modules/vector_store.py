from uuid import uuid4
from langchain_community.vectorstores import Chroma
from .embeddings import SentenceTransformerEmbeddings

class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "default"):
        """Initialize the vector store."""
        self.embedding_function = SentenceTransformerEmbeddings('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        self.vectordb = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function
        )

    def add_documents(self, chunks: list):
        """Add documents to the vector store."""
        ids = [str(uuid4()) for _ in range(len(chunks))]
        self.vectordb.add_texts(
            texts=chunks,
            ids=ids
        )

    def similarity_search(self, query: str, k: int = 3):
        """Perform similarity search."""
        return self.vectordb.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs=None):
        """Get the vector store as a retriever."""
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.vectordb.as_retriever(search_kwargs=search_kwargs) 
