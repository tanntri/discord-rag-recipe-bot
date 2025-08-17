import shutil # Import shutil for directory deletion
import hashlib
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from utils.llm import EmbeddingModel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PaginatedPipelineOptions
)
from docling.document_converter import DocumentConverter, WordFormatOption
from pathlib import Path

class VectorStore:
    def __init__(self, persist_directory: str = "db"):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.doc_path = base_dir / "data" / "personal_recipe.docx"
        self.persist_directory = base_dir / persist_directory
        self.embedding_model = EmbeddingModel().get_embedding_model()
        self.vector_store = self.load_or_create_vector_store()
        self.metadata_file = self.persist_directory / "last_hash.txt"

    def _get_file_hash(self) -> str:
        """Generates an MD5 hash of the document file."""
        hasher = hashlib.md5()
        with open(self.doc_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _delete_vector_store(self):
        """Deletes the existing vector store directory."""
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            print("Existing vector store deleted.")

    def convert_to_markdown(self) -> str:
        """ Convert DOCX document to markdown text """
        pipeline_options = PaginatedPipelineOptions()
        converter = DocumentConverter(
            format_options={
                InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(self.doc_path)
        document = result.document
        text = document.export_to_markdown()
        return text

    def load_or_create_vector_store(self):
        # A new check is performed before creating or loading the store.
        if self.persist_directory.exists():
            current_hash = self._get_file_hash()
            metadata_file = self.persist_directory / "last_hash.txt"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    saved_hash = f.read().strip()
                
                # If the hash matches, the vector store is up-to-date.
                if saved_hash == current_hash:
                    print("Vector store is up-to-date. Loading existing database.")
                    return Chroma(
                        persist_directory=str(self.persist_directory),
                        embedding_function=self.embedding_model,
                        collection_name="recipes"
                    )
            
            # If hashes don't match or the metadata file doesn't exist, rebuild.
            print("Document content has changed. Rebuilding vector store.")
            self._delete_vector_store()

        print("Creating new vector store.")
        try:
            documents = self.convert_to_markdown()
        except Exception as e:
            raise FileNotFoundError(f"Error converting document: {e}")
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(documents)

        if not md_header_splits:
            print("Warning: Document splitting resulted in no chunks. Check document formatting.")
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            md_header_splits = recursive_splitter.split_text(documents)

        vector_store = Chroma.from_documents(
            documents=md_header_splits,
            persist_directory=str(self.persist_directory),
            embedding=self.embedding_model,
            collection_name="recipes"
        )
        
        vector_store.persist()
        
        # Save the new hash to the metadata file
        metadata_file = self.persist_directory / "last_hash.txt"
        with open(metadata_file, 'w') as f:
            f.write(self._get_file_hash())

        return vector_store
    
    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    
if __name__ == "__main__":
    vector_store_instance = VectorStore()
    retriever = vector_store_instance.get_retriever()
    print("Vector store initialized and ready.")
    query = "What are the ingredients for Mapo Tofu?"
    docs = retriever.invoke(query)
    print(docs)