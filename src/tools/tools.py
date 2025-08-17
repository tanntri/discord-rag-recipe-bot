from langchain_core.tools import tool
from utils.vector import VectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent
dotenv_path = base_dir / ".env"

load_dotenv(dotenv_path=dotenv_path)
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

vector_store_instance = VectorStore()
retriever = vector_store_instance.get_retriever()

@tool
def retriever_tool(query: str) -> str:
    """Tool to retrieve relevant documents based on a query."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def search_tool(query: str):
    """ perform a web search using Tavily to answer questions from the query """
    tavily = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)
    result = tavily.invoke(query)
    return result

if __name__ == "__main__":
    sample_query = "What are the ingredients for Mapo Tofu?"
    # retrieved_content = retriever_tool.invoke(sample_query)
    search_result = search_tool.invoke(sample_query)
    print("Retrieved Documents:")
    # print(retrieved_content)
    print("Search Results:")
    print(search_result)