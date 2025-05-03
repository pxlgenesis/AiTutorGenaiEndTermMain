# src/config.py
import os
from dotenv import load_dotenv
import logging
import sys

# --- Basic Logging Setup ---
# Configure logging early
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to stdout

# --- Load Environment Variables ---
# Construct the path to the .env file relative to this config file's location
# config.py is in src/, .env is one level up
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
     logging.warning(f".env file not found at expected location: {dotenv_path}. Relying on environment variables.")
load_dotenv(dotenv_path=dotenv_path)
logging.info(f"Attempted loading environment variables from: {dotenv_path}")

# --- Paths Configuration ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) # Root is one level above src/
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
INPUT_DATA_PATH = os.path.join(DATA_DIR, "input", "course_data")
OUTPUT_DATA_PATH = os.path.join(DATA_DIR, "output")
VECTOR_DB_PATH = os.path.join(OUTPUT_DATA_PATH, "vector_db_chroma")
SOURCE_INFO_FILE = os.path.join(OUTPUT_DATA_PATH, "source_documents.txt")
logging.info(f"Project Root: {_PROJECT_ROOT}")
logging.info(f"Input Data Path: {INPUT_DATA_PATH}")
logging.info(f"Vector DB Path: {VECTOR_DB_PATH}")

# --- Embeddings Configuration (Using Google consistently here) ---
EMBEDDING_MODEL_PROVIDER = "google" # Keep this consistent unless changing embedding strategy
EMBEDDING_MODEL_NAME = "models/embedding-001"
logging.info(f"Using Embedding Provider: {EMBEDDING_MODEL_PROVIDER}, Model: {EMBEDDING_MODEL_NAME}")

# --- LLM Providers & Models Configuration ---
# Define the available LLMs with user-facing names as keys
AVAILABLE_LLMS = {
    # User-facing Name: { internal_provider_id, model_api_name, env_var_for_key, default_temp }
    "Gemini Flash": {
        "provider": "google",
        "model_name": "gemini-2.0-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "temperature": 0.3,
    },
    "OpenAI GPT-3.5": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo", # Consider updating to gpt-4o if preferred/available
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.3,
    },
    "Mistral Small": {
        "provider": "mistral",
        "model_name": "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "temperature": 0.3,
    }
    # Add other models here if needed in the future
}
logging.info(f"Configured LLMs: {list(AVAILABLE_LLMS.keys())}")

# --- API Key Loading and Provider Availability Check ---
API_KEYS = {} # Stores loaded keys mapped by internal provider ID ('google', 'openai', 'mistral')
AVAILABLE_PROVIDERS = [] # Stores user-facing names of providers for which keys were found

for user_facing_name, config_details in AVAILABLE_LLMS.items():
    env_var_name = config_details["api_key_env"]
    provider_id = config_details["provider"]
    api_key = os.getenv(env_var_name)

    if api_key:
        API_KEYS[provider_id] = api_key # Store key by internal provider ID
        AVAILABLE_PROVIDERS.append(user_facing_name) # Add user-facing name to available list
        logging.info(f"API Key found for {user_facing_name} (Provider ID: {provider_id}).")
    else:
        logging.warning(f"API Key environment variable '{env_var_name}' not found for {user_facing_name}. This model will be unavailable.")

# --- Default LLM Selection ---
# Try to set a sensible default based on availability, preferring Gemini
DEFAULT_LLM_PROVIDER_NAME = None
if "Gemini Flash" in AVAILABLE_PROVIDERS:
    DEFAULT_LLM_PROVIDER_NAME = "Gemini Flash"
elif AVAILABLE_PROVIDERS: # If Gemini not available, pick the first one that is
     DEFAULT_LLM_PROVIDER_NAME = AVAILABLE_PROVIDERS[0]
logging.info(f"Default LLM Provider set to: {DEFAULT_LLM_PROVIDER_NAME}")

# --- RAG Specific Settings ---
RETRIEVER_K = 3 # Number of chunks to retrieve for context
CHUNK_SIZE = 1000 # Target size for text chunks during ingestion
CHUNK_OVERLAP = 200 # Overlap between consecutive chunks
logging.info(f"RAG settings: Retriever K={RETRIEVER_K}, Chunk Size={CHUNK_SIZE}, Chunk Overlap={CHUNK_OVERLAP}")

# --- Critical Validations ---
# 1. Check if embedding provider's key is available
embedding_key_needed = API_KEYS.get(EMBEDDING_MODEL_PROVIDER)
if not embedding_key_needed:
     # Specifically check Google key if it's the embedding provider
     if EMBEDDING_MODEL_PROVIDER == "google" and not API_KEYS.get("google"):
         logging.error(f"FATAL: Google API key (GOOGLE_API_KEY) required for embeddings ('{EMBEDDING_MODEL_NAME}') but not found. Set it in .env or environment.")
         # In a production scenario, you might raise an exception or exit here
         # raise ValueError("Missing API key required for embeddings.")
     elif EMBEDDING_MODEL_PROVIDER != "google": # Check if a non-google embedding provider was intended but key is missing
          logging.error(f"FATAL: API key for embedding provider '{EMBEDDING_MODEL_PROVIDER}' not found.")
          # raise ValueError(f"Missing API key required for embeddings provider: {EMBEDDING_MODEL_PROVIDER}")

# 2. Check if *any* LLM provider is available
if not AVAILABLE_PROVIDERS:
    logging.error("FATAL: No API Keys found or loaded for any configured LLM provider (Gemini, OpenAI, Mistral). The application cannot function.")
    # raise ValueError("No LLM providers available due to missing API keys.")

logging.info("Configuration loading completed.")

# --- Optional: Ensure Directories Exist ---
# Can be useful, but might also be handled by scripts or other parts of the code
# try:
#     os.makedirs(INPUT_DATA_PATH, exist_ok=True)
#     os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
#     os.makedirs(VECTOR_DB_PATH, exist_ok=True)
#     logging.info("Ensured data directories exist.")
# except OSError as e:
#     logging.error(f"Error creating data directories: {e}")