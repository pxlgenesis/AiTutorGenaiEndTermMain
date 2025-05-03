# src/ingestion/ingest.py

import os
import glob
import logging
import sys

# Third-party imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader, # Example: Add loader for .docx
    # Add other loaders as needed (CSVLoader, UnstructuredExcelLoader, etc.)
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Langchain Imports for Embeddings (Dynamically load based on config)
# We import the specific class needed based on the config later,
# but it's good practice to know which ones *could* be used.
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings # Example

# Local application imports
from src import config # Import configuration constants

# Setup logger for this module
logger = logging.getLogger(__name__)
# Configure logging if run as main script (or rely on root config)
if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# --- Document Loading Configuration ---
# Define loaders for different file types
# Add or remove loaders based on the document types you need to support
LOADER_MAPPING = {
    ".pdf": {"loader_cls": PyPDFLoader, "loader_kwargs": {}},
    ".txt": {"loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}},
    ".doc": {"loader_cls": UnstructuredWordDocumentLoader, "loader_kwargs": {}},
    ".docx": {"loader_cls": UnstructuredWordDocumentLoader, "loader_kwargs": {}},
    # Add mappings for .csv, .xlsx, .html, etc. if needed
    # ".csv": {"loader_cls": CSVLoader, "loader_kwargs": {"encoding": "utf-8"}},
    # ".xlsx": {"loader_cls": UnstructuredExcelLoader, "loader_kwargs": {}},
}

def load_documents(source_dir: str) -> list:
    """
    Loads documents from the source directory using configured loaders.

    Args:
        source_dir: The path to the directory containing source documents.

    Returns:
        A list of loaded LangChain Document objects.
    """
    loaded_docs = []
    all_files = []
    logger.info(f"Scanning for documents in: {source_dir}")

    # Find all files matching the loader patterns
    for ext in LOADER_MAPPING:
        pattern = f"**/*{ext}"
        found = glob.glob(os.path.join(source_dir, pattern), recursive=True)
        all_files.extend(found)
        logger.info(f"Found {len(found)} files matching pattern *{ext}")

    if not all_files:
        logger.warning(f"No documents found in {source_dir} matching configured loader types: {list(LOADER_MAPPING.keys())}")
        return []

    # Record found files for user reference
    try:
        # Ensure output directory exists before writing the file list
        os.makedirs(config.OUTPUT_DATA_PATH, exist_ok=True)
        with open(config.SOURCE_INFO_FILE, 'w', encoding='utf-8') as f:
            for filepath in sorted(all_files):
                f.write(os.path.basename(filepath) + '\n')
        logger.info(f"Saved list of {len(all_files)} found source documents to {config.SOURCE_INFO_FILE}")
    except Exception as e:
        logger.error(f"Could not write source document list to {config.SOURCE_INFO_FILE}: {e}")


    # Load documents using the appropriate loader
    for file_path in all_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in LOADER_MAPPING:
            loader_info = LOADER_MAPPING[file_ext]
            loader_cls = loader_info["loader_cls"]
            loader_kwargs = loader_info["loader_kwargs"]
            logger.debug(f"Loading document: {file_path} using {loader_cls.__name__}")
            try:
                loader = loader_cls(file_path, **loader_kwargs)
                docs = loader.load() # Load returns a list of Documents
                # Add source metadata consistently
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path) # Use filename as source base
                loaded_docs.extend(docs)
                logger.debug(f"Successfully loaded {len(docs)} document pages/sections from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error loading document {file_path} with {loader_cls.__name__}: {e}", exc_info=True)
        else:
            # Should not happen if all_files logic is correct, but good failsafe
            logger.warning(f"Skipping file with unmapped extension: {file_path}")

    logger.info(f"Total documents loaded: {len(loaded_docs)} pages/sections from {len(set(d.metadata.get('source', '') for d in loaded_docs))} files.")
    return loaded_docs


def split_text(documents: list) -> list:
    """
    Splits loaded documents into smaller chunks.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        A list of smaller Document chunks.
    """
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helpful for potential future context window management
    )
    try:
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
        return split_docs
    except Exception as e:
        logger.error(f"Error during text splitting: {e}", exc_info=True)
        # Depending on severity, you might want to re-raise or exit
        raise # Re-raise the exception to halt ingestion

def add_citation_metadata(documents: list) -> list:
    """
    Adds a 'citation' field to each document chunk's metadata.

    Args:
        documents: A list of Document chunks.

    Returns:
        The list of Document chunks with added 'citation' metadata.
    """
    logger.info("Adding citation metadata to chunks...")
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown Source') # Already set in load_documents
        page = doc.metadata.get('page', None) # PyPDFLoader adds this

        # Format page number nicely if available
        page_num_display = f"page {page + 1}" if isinstance(page, int) else None

        # Construct citation string
        citation_parts = [source]
        if page_num_display:
            citation_parts.append(page_num_display)
        # Add other metadata if needed, e.g., section, paragraph id

        doc.metadata['citation'] = ", ".join(filter(None, citation_parts)) # Join parts that exist
        logger.debug(f"Added citation '{doc.metadata['citation']}' to chunk starting with: '{doc.page_content[:50]}...'")
    return documents

def get_embedding_function():
    """
    Dynamically loads the embedding function based on configuration.
    Handles potential import errors and missing API keys.
    """
    provider = config.EMBEDDING_MODEL_PROVIDER
    model_name = config.EMBEDDING_MODEL_NAME
    api_key = config.API_KEYS.get(provider)

    logger.info(f"Initializing embedding function for provider: {provider}, model: {model_name}")

    if not api_key:
         # Special check for Google as it's common
         if provider == "google":
              google_api_key = config.API_KEYS.get("google")
              if google_api_key:
                   api_key = google_api_key
              else:
                   logger.error("FATAL: Google API key (GOOGLE_API_KEY) required for Google embeddings but not found.")
                   raise ValueError("Missing GOOGLE_API_KEY for embeddings")
         else:
              logger.error(f"FATAL: API Key for embedding provider '{provider}' not found in environment/config.")
              raise ValueError(f"Missing API key for embeddings provider: {provider}")

    try:
        if provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            logger.info(f"Using Google Embeddings: {model_name}")
            return embeddings
        # Add elif blocks for other providers (e.g., OpenAI, HuggingFace) if needed
        # elif provider == "openai":
        #     from langchain_openai import OpenAIEmbeddings
        #     # Ensure OPENAI_API_KEY is handled in config.API_KEYS if using this
        #     embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
        #     logger.info(f"Using OpenAI Embeddings: {model_name}")
        #     return embeddings
        else:
            logger.error(f"FATAL: Unsupported embedding provider specified in config: {provider}")
            raise NotImplementedError(f"Embedding provider '{provider}' not supported.")

    except ImportError as e:
         logger.error(f"FATAL: Failed to import embedding library for {provider}. Did you install necessary packages? Error: {e}")
         raise ImportError(f"Missing library for {provider} embeddings.") from e
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize {provider} Embeddings model '{model_name}': {e}", exc_info=True)
        # Re-raise to indicate critical failure
        raise RuntimeError(f"Could not initialize embedding model: {e}") from e


def create_and_persist_vector_store(documents: list, embeddings):
    """
    Creates a Chroma vector store from documents and persists it.

    Args:
        documents: List of document chunks with metadata.
        embeddings: The initialized embedding function.
    """
    vector_db_path = config.VECTOR_DB_PATH
    logger.info(f"Creating/updating vector store at: {vector_db_path} using {len(documents)} chunks...")
    logger.info(f"Using embedding model: {getattr(embeddings, 'model', 'N/A')} (Class: {embeddings.__class__.__name__})") # Log model info

    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)

        # Create Chroma DB from documents
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vector_db_path,
            # Consider collection metadata if useful:
            # collection_metadata={"hnsw:space": "cosine"} # Example, check Chroma docs
        )

        # Persist the database to disk
        logger.info("Persisting vector store...")
        vector_store.persist()
        logger.info(f"Vector store created/updated and persisted successfully at: {vector_db_path}")

    except Exception as e:
        logger.error(f"FATAL: Failed to create or persist vector store: {e}", exc_info=True)
        # Indicate critical failure
        raise RuntimeError("Failed to create/persist vector store.") from e


# --- Main Ingestion Pipeline Execution ---
def main():
    """Runs the complete data ingestion pipeline."""
    logger.info("--- Starting Data Ingestion Pipeline ---")

    # 1. Load Documents
    try:
        loaded_documents = load_documents(config.INPUT_DATA_PATH)
        if not loaded_documents:
            logger.warning("No documents were loaded. Exiting ingestion pipeline.")
            return # Exit gracefully if no documents found/loaded
    except Exception as e:
         logger.error(f"Critical error during document loading: {e}", exc_info=True)
         return # Stop pipeline if loading fails critically

    # 2. Split Documents
    try:
        split_docs = split_text(loaded_documents)
        if not split_docs:
             logger.error("Text splitting resulted in zero chunks. Cannot proceed.")
             return
    except Exception as e:
         logger.error(f"Critical error during text splitting: {e}. Aborting.", exc_info=True)
         return # Stop if splitting fails

    # 3. Add Citation Metadata
    try:
         docs_with_citations = add_citation_metadata(split_docs)
    except Exception as e:
         logger.error(f"Error adding citation metadata: {e}. Proceeding without citations potentially affected.", exc_info=True)
         docs_with_citations = split_docs # Fallback to docs without citations if needed, or handle differently

    # 4. Initialize Embedding Function
    try:
        embeddings = get_embedding_function()
    except (ValueError, ImportError, NotImplementedError, RuntimeError) as e:
         logger.error(f"Failed to initialize embedding function: {e}. Ingestion cannot continue.")
         return # Cannot proceed without embeddings

    # 5. Create and Persist Vector Store
    try:
        create_and_persist_vector_store(docs_with_citations, embeddings)
    except RuntimeError as e:
         logger.error(f"Failed to create/persist vector store: {e}. Ingestion failed.")
         return # Stop if DB creation fails

    logger.info("--- Data Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    # This block executes only when the script is run directly (e.g., python src/ingestion/ingest.py)
    main()