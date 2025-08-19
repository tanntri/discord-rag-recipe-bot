import os
import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever as LangChainVectorStoreRetriever
from dotenv import load_dotenv
from utils.llm import EmbeddingModel

load_dotenv()

class VectorStore:
    """
    Connects to a remote Chroma Cloud collection and returns a retriever.
    This class is intended for a live web service and assumes the
    vector store is already populated by a separate ingestion pipeline.
    """

    def __init__(self):
        # Environment variables for configuration
        # self.chroma_host = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "recipes")
        self.chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")
        self.chroma_database = os.getenv("CHROMA_DATABASE", "default_database")

        # Raise an error if a critical environment variable is missing
        if not self.chroma_api_key:
            raise EnvironmentError("CHROMA_API_KEY environment variable is required.")

        # Initialize the embedding model, which is used for querying the vector store
        self.embedding_model = EmbeddingModel().get_embedding_model()

        # Initialize the Chroma Cloud client
        self.chroma_client = chromadb.CloudClient(
            api_key=self.chroma_api_key,
            tenant=self.chroma_tenant,
            database=self.chroma_database
        )

        # Load the vector store from the cloud upon initialization
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        """
        Load the Chroma vector store from the cloud using the HTTP client.
        
        Returns:
            Chroma: The initialized Chroma vector store.
        """
        print(f"Loading Chroma collection '{self.collection_name}' from cloud")
        return Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )

    def get_retriever(self) -> LangChainVectorStoreRetriever:
        """
        Return a retriever configured for similarity search.
        
        Returns:
            LangChainVectorStoreRetriever: The configured retriever instance.
        """
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})


if __name__ == "__main__":
    # This block allows you to manually test the class as a standalone script
    try:
        # Initialize the vector store and get the retriever
        vs = VectorStore()
        retriever = vs.get_retriever()
        print("Successfully initialized VectorStore and got retriever.")

        # Example query
        query = "What are the ingredients for Mapo Tofu?"
        docs = retriever.invoke(query)
        print("Query results:", docs)

    except Exception as e:
        print(f"An error occurred: {e}")
