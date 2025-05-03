# test_retriever.py
import os
import sys
import logging
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Add src to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    logger.info(f"Added src directory to sys.path: {SRC_DIR}")

# --- Imports (after path setup) ---
try:
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from src import config # Load config to get paths and keys
except ImportError as e:
    logger.error(f"Failed to import necessary libraries: {e}")
    logger.error("Make sure requirements are installed in the active environment.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error during initial imports: {e}")
    sys.exit(1)

# --- Main Test Logic ---
def run_test():
    logger.info("--- Starting Retriever Test ---")

    # 1. Get API Key from config dictionary
    google_api_key = config.API_KEYS.get('google') # <-- FIX: Access key from dict
    if not google_api_key:                         # <-- FIX: Check if key exists in dict
        logger.error("Google API Key not found in config.API_KEYS['google']. Check .env and config.py loading.")
        return False
    logger.info("Found Google API Key in config.")

    # 2. Initialize Embeddings
    logger.info(f"Initializing embeddings: {config.EMBEDDING_MODEL_NAME}")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            google_api_key=google_api_key # <-- FIX: Use the key retrieved from dict
        )
        logger.info("Embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        return False

    # 3. Check if Vector DB Path Exists
    vector_db_path = config.VECTOR_DB_PATH
    if not os.path.exists(vector_db_path):
        logger.error(f"Vector DB directory not found at: {vector_db_path}")
        logger.error("Please run the data ingestion first.")
        return False
    logger.info(f"Vector DB path found: {vector_db_path}")

    # 4. Load Chroma Vector Store and Retriever
    logger.info("Attempting to load Chroma vector store...")
    try:
        vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings
        )
        logger.info("Chroma vector store loaded.")
        logger.info("Attempting to get retriever...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        logger.info("Retriever created successfully.")
    except Exception as e:
        logger.error(f"Failed to load vector store or create retriever: {e}", exc_info=True)
        return False

    # 5. Test Retrieval (Simple Query)
    test_query = "What is cognitive impairment?" # Use a generic query
    logger.info(f"Attempting simple retrieval with query: '{test_query}'")
    try:
        results = retriever.invoke(test_query)
        logger.info(f"Retrieval successful. Found {len(results)} documents.")
        if results:
            logger.info("First result metadata: " + str(results[0].metadata))
            logger.info("First result snippet: " + results[0].page_content[:150] + "...")
        else:
             logger.warning("Query returned no results (this might be okay depending on content).")
    except Exception as e:
        logger.error(f"Failed during retriever invocation: {e}", exc_info=True)
        return False

    logger.info("--- Retriever Test Completed Successfully ---")
    return True

if __name__ == "__main__":
    if run_test():
        logger.info("Test script finished OK.")
        print("\nTest OK. Press Enter to exit.")
        input()
        sys.exit(0)
    else:
        logger.error("Test script FAILED.")
        print("\nTest FAILED. Check logs above. Press Enter to exit.")
        input()
        sys.exit(1)
        