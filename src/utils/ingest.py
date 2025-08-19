import os
import hashlib
import chromadb
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from utils.llm import EmbeddingModel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.document_converter import DocumentConverter, WordFormatOption
from dotenv import load_dotenv

load_dotenv()


def get_file_hash(file_path: Path) -> str:
    """
    Generate an MD5 hash of a file's contents.
    
    Args:
        file_path (Path): The path to the file.
        
    Returns:
        str: The MD5 hash of the file.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def convert_docx_to_markdown(doc_path: Path) -> str:
    """
    Convert a DOCX document to a markdown string.
    
    Args:
        doc_path (Path): The path to the DOCX file.
    
    Returns:
        str: The markdown content of the document.
    """
    print(f"Converting DOCX: {doc_path}")
    pipeline_options = PaginatedPipelineOptions()
    converter = DocumentConverter(
        format_options={
            InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(doc_path)
    document = result.document
    return document.export_to_markdown()


def split_markdown(markdown_text: str, file_hash: str):
    """
    Split markdown into chunks and add file hash to metadata.
    
    Args:
        markdown_text (str): The markdown content to split.
        file_hash (str): The hash of the source file.
        
    Returns:
        List[Document]: A list of document chunks with metadata.
    """
    print("Splitting markdown into chunks...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = splitter.split_text(markdown_text)

    if not splits:
        print("Markdown splitter failed, using RecursiveCharacterTextSplitter.")
        recursive = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = recursive.split_text(markdown_text)
    
    # Add the file hash to the metadata of each chunk
    for doc in splits:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["file_hash"] = file_hash

    return splits


def ingest_to_chroma_cloud(markdown_chunks, collection_name: str, embedding_model, chroma_host: str, chroma_api_key: str, chroma_tenant: str, chroma_database: str, file_hash: str):
    """
    Ingests document chunks into a Chroma Cloud collection only if the file hash has changed.
    
    Args:
        markdown_chunks (List[Document]): The document chunks to ingest.
        collection_name (str): The name of the Chroma collection.
        embedding_model (EmbeddingFunction): The embedding model to use.
        chroma_host (str): The Chroma Cloud host address.
        chroma_api_key (str): The API key for authentication.
        file_hash (str): The hash of the source file.
    """
    print(f"Connecting to Chroma Cloud at {chroma_host}...")
    chroma_client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database
    )

    
    # Check for an existing collection and the presence of the current file hash
    try:
        if collection_name in [c.name for c in chroma_client.list_collections()]:
            collection = chroma_client.get_collection(name=collection_name)
            # Use the where clause to find if any document has the current hash
            results = collection.get(where={"file_hash": file_hash})
            if results["ids"]:
                print(f"Collection '{collection_name}' already contains content with this hash. Skipping ingestion.")
                return
    except Exception as e:
        print(f"An error occurred while checking the collection: {e}")
        # Proceed with ingestion if the check fails

    print(f"Creating or updating collection '{collection_name}'...")
    Chroma.from_documents(
        documents=markdown_chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        client=chroma_client
    )
    print("Successfully ingested data into Chroma Cloud.")


if __name__ == "__main__":
    print("Starting ingestion process...")

    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    print(f"Base directory: {base_dir}")
    doc_path = base_dir / "data" / "personal_recipe.docx"

    if not doc_path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")

    # Environment variables
    chroma_host = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
    chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")
    chroma_database = os.getenv("CHROMA_DATABASE", "default_database")
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "recipes")

    if not chroma_api_key:
        raise EnvironmentError("CHROMA_API_KEY must be set in your environment.")

    # Get the file's hash before starting the pipeline
    file_hash = get_file_hash(doc_path)

    # Start pipeline
    markdown_text = convert_docx_to_markdown(doc_path)
    markdown_chunks = split_markdown(markdown_text, file_hash)

    embedding_model = EmbeddingModel().get_embedding_model()

    ingest_to_chroma_cloud(
        markdown_chunks,
        collection_name,
        embedding_model,
        chroma_host,
        chroma_api_key,
        chroma_tenant,
        chroma_database,
        file_hash
    )
