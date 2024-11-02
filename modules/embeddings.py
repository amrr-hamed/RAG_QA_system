from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        """Embed a single query (for retrieval)."""
        return self.model.encode(text)

    def embed_documents(self, texts: list):
        """Embed multiple documents (for storing in the vector store)."""
        return self.model.encode(texts)
