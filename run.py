# run.py
import os
import sys
import subprocess
import importlib.util
import argparse
import platform
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
APP_FILE_PATH = os.path.join(SRC_DIR, "ui", "app.py") # Path to the actual Streamlit app
INGEST_MODULE_PATH = os.path.join(SRC_DIR, "ingestion", "ingest.py") # Path to the ingestion script
VENV_CHECK_VAR = 'VIRTUAL_ENV'

# --- Helper Functions ---

def check_virtual_env():
    """Checks if a virtual environment appears to be active and warns if not."""
    venv_path = os.getenv(VENV_CHECK_VAR)
    is_venv = sys.prefix != sys.base_prefix # More reliable check

    if not is_venv:
        logger.warning("-" * 60)
        logger.warning("WARNING: Not running in a detected virtual environment.")
        logger.warning("It is STRONGLY recommended to use a virtual environment (Python 3.10 or 3.11).")
        logger.warning("Create one using: python -m venv venv_py310 (replace python with py -3.10 if needed)")
        if platform.system() == "Windows":
            logger.warning("Activate using:   .\\venv_py310\\Scripts\\activate.bat (cmd)")
            logger.warning("                .\\venv_py310\\Scripts\\Activate.ps1 (PowerShell)")
        else: # Linux/macOS
            logger.warning("Activate using:   source venv_py310/bin/activate")
        logger.warning("-" * 60)
    else:
        logger.info(f"Virtual environment appears active: {sys.prefix}")
    return is_venv

def setup_python_path():
    """Adds project root and src directories to sys.path for imports."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        logger.info(f"Added project root to sys.path: {PROJECT_ROOT}")
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
        logger.info(f"Added src directory to sys.path: {SRC_DIR}")

def run_ingestion():
    """Imports and runs the main() function from the ingestion script."""
    logger.info("--- Running Data Ingestion ---")
    try:
        # Ensure config can be imported after path setup
        from src import config # Now safe to import
        # Dynamically import the ingest module
        module_name = "src.ingestion.ingest"
        spec = importlib.util.spec_from_file_location(module_name, INGEST_MODULE_PATH)
        if spec is None or spec.loader is None:
            logger.error(f"Could not create module spec for {INGEST_MODULE_PATH}")
            return False
        ingest_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = ingest_module
        spec.loader.exec_module(ingest_module)
        logger.info(f"Successfully imported ingestion module: {module_name}")

        # Call the main function if it exists
        if hasattr(ingest_module, 'main') and callable(ingest_module.main):
            logger.info("Calling main() function in ingestion module...")
            ingest_module.main()
            logger.info("--- Data Ingestion Complete ---")
            return True
        else:
            logger.error(f"Could not find a callable 'main' function in {INGEST_MODULE_PATH}")
            return False
    except ImportError as e:
        logger.exception(f"Failed to import ingestion module or its dependencies. Ensure requirements are installed in the active venv.")
        return False
    except Exception as e:
        logger.exception(f"An exception occurred during data ingestion:")
        return False

def run_streamlit_app():
    """Launches the Streamlit app using 'streamlit run' via subprocess."""
    logger.info("--- Starting Streamlit Application ---")
    # Construct the command using sys.executable to ensure correct python environment
    streamlit_command = [
        sys.executable, # The python.exe from the active (hopefully venv) environment
        "-m",
        "streamlit",
        "run",
        APP_FILE_PATH, # The path to src/ui/app.py
        "--server.port", "8501",
        "--server.address", "0.0.0.0" # Allows network access
    ]
    logger.info(f"Executing command: {' '.join(streamlit_command)}")
    try:
        # Start the Streamlit process
        process = subprocess.run(streamlit_command, check=False) # Don't raise error on non-zero exit (like Ctrl+C)
        logger.info(f"--- Streamlit App Stopped (Exit Code: {process.returncode}) ---")
        return process.returncode == 0 # True if stopped cleanly
    except FileNotFoundError:
        logger.error(f"ERROR: Failed to run Streamlit. Is '{sys.executable} -m streamlit' valid?")
        logger.error("Make sure Streamlit and other requirements are installed in your active Python environment ('pip install -r requirements.txt').")
        return False
    except KeyboardInterrupt:
        logger.info("--- Streamlit App Interrupted by User (Ctrl+C) ---")
        return True # Treat Ctrl+C as normal stop
    except Exception as e:
        logger.exception(f"An unexpected error occurred while launching or running Streamlit:")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Tutor Bot ingestion and Streamlit app.")
    parser.add_argument( "--skip-ingest", action="store_true", help="Skip data ingestion.")
    args = parser.parse_args()

    logger.info("============================================")
    logger.info(" AI Tutor Bot Startup Script (run.py) ")
    logger.info("============================================")

    # 1. Check for Venv (Recommended)
    check_virtual_env()

    # 2. Set up Python Path
    setup_python_path()

    # 3. Run Ingestion (Optional)
    ingestion_success = True
    if not args.skip_ingest:
        ingestion_success = run_ingestion()
    else:
        logger.info("--- Skipping Data Ingestion (--skip-ingest flag detected) ---")

    # 4. Run Streamlit App (if ingestion was successful or skipped)
    if ingestion_success:
        run_streamlit_app() # This function executes 'streamlit run ...'
    else:
        logger.error("--- Streamlit application will not start due to errors during data ingestion. ---")
        sys.exit(1) # Exit with error status

    logger.info("--- Script Finished ---")
    sys.exit(0) # Exit successfully