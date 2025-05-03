# src/pipeline/rag.py

import logging
import os
import sys
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

try:
    from src.utils.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    logger.info("Using basic logging setup for rag.py.")

try:
    from src import config
except ImportError:
    logger.critical("FATAL: Cannot import configuration from src.config. Cannot proceed.")
    st.error("FATAL: Cannot import configuration from src.config.")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_function():
    provider = config.EMBEDDING_MODEL_PROVIDER
    model_name = config.EMBEDDING_MODEL_NAME
    logger.info(f"Attempting to load embedding function: Provider='{provider}', Model='{model_name}'")

    api_key = config.API_KEYS.get(provider)
    if not api_key and provider == "google":
         api_key = config.API_KEYS.get("google")

    if not api_key:
        error_msg = f"API Key for embedding provider '{provider}' not found. Cannot load embeddings."
        logger.error(error_msg)
        st.error(error_msg)
        return None

    try:
        if provider == "google":
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            logger.info(f"Successfully loaded Google Embeddings: {model_name}")
            return embeddings
        else:
            error_msg = f"Unsupported embedding provider configured: {provider}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    except ImportError as e:
         error_msg = f"Failed to import embedding library for {provider}. Install required packages? Error: {e}"
         logger.error(error_msg, exc_info=True)
         st.error(f"Missing library for {provider} embeddings. Check requirements.txt.")
         return None
    except Exception as e:
        error_msg = f"Failed to initialize {provider} Embeddings model '{model_name}': {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"Could not initialize embedding model: {error_msg}")
        return None


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever(_embedding_function, k: int):
    vector_db_path = config.VECTOR_DB_PATH
    logger.info(f"Attempting to load vector store from: {vector_db_path} with k={k}")

    if not _embedding_function:
        logger.error("Cannot load retriever: Embedding function is not available.")
        st.error("Knowledge base requires embeddings, which failed to load.")
        return None

    if not os.path.exists(vector_db_path):
        error_msg = f"Vector DB not found at '{vector_db_path}'. Did you run the ingestion script?"
        logger.error(error_msg)
        st.error(error_msg)
        return None

    try:
        vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=_embedding_function
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        logger.info(f"Retriever loaded successfully. Configured to retrieve top {k} chunks.")
        return retriever
    except Exception as e:
        error_msg = f"Error loading vector store from {vector_db_path}: {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"Error loading knowledge base: {e}")
        return None


@st.cache_resource(show_spinner="Initializing language model...")
def load_llm(provider_name: str, temperature: float):
    logger.info(f"Attempting to load LLM: Provider='{provider_name}', Temperature={temperature}")

    if not provider_name or provider_name not in config.AVAILABLE_LLMS:
        error_msg = f"Invalid or unavailable LLM provider specified: '{provider_name}'."
        logger.error(error_msg)
        return None

    llm_config = config.AVAILABLE_LLMS[provider_name]
    provider_id = llm_config["provider"]
    model_api_name = llm_config["model_name"]
    api_key = config.API_KEYS.get(provider_id)

    if not api_key:
        error_msg = f"API Key for provider '{provider_id}' (required by {provider_name}) not found. Check .env/config."
        logger.error(error_msg)
        st.error(f"Configuration Error: Missing API Key for {provider_name}.")
        return None

    logger.info(f"Initializing LLM: Name='{provider_name}', ProviderID='{provider_id}', Model='{model_api_name}', Temp={temperature}")
    try:
        if provider_id == "google":
            llm = ChatGoogleGenerativeAI(
                model=model_api_name,
                temperature=temperature,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
        elif provider_id == "openai":
            llm = ChatOpenAI(
                model=model_api_name,
                temperature=temperature,
                openai_api_key=api_key
            )
        elif provider_id == "mistral":
             llm = ChatMistralAI(
                model=model_api_name,
                temperature=temperature,
                mistral_api_key=api_key
             )
        else:
            error_msg = f"Internal configuration error: Unknown LLM provider ID '{provider_id}' for {provider_name}."
            logger.error(error_msg)
            st.error(error_msg)
            return None

        logger.info(f"LLM '{provider_name}' initialized successfully (Class: {llm.__class__.__name__}).")
        return llm
    except ImportError as e:
         error_msg = f"Failed to import library for {provider_id} LLM. Install required packages? Error: {e}"
         logger.error(error_msg, exc_info=True)
         st.error(f"Missing library for {provider_name}. Check requirements.txt.")
         return None
    except Exception as e:
        error_msg = f"Failed to initialize LLM '{provider_name}' (Model: {model_api_name}): {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"Could not initialize Language Model '{provider_name}': Check API key validity and model access.")
        return None


RAG_PROMPT_TEMPLATE = """
**Role:** You are an AI assistant acting as a helpful tutor for a specific course module.

**Constraint:** Your knowledge is STRICTLY LIMITED to the text provided in the 'Context' section below. Do NOT use any prior knowledge or information outside of this context.

**Task:** Answer the student's 'Question' accurately and concisely using ONLY the provided 'Context'.

**Instructions:**
1. Carefully analyze the 'Context' sections provided. Each section includes a 'Citation' indicating its source.
2. Formulate an answer that directly addresses the 'Question'.
3. Base your entire answer **exclusively** on information found within the 'Context'.
4. If the 'Context' does not contain information relevant to the 'Question', you MUST state: "Based on the provided documents, I cannot answer this question." Do not guess or synthesize information.
5. After providing the answer (or stating you cannot answer), list **all** the source 'Citation'(s) for the context chunks you consulted to arrive at your conclusion. Format citations clearly (e.g., "Sources Consulted: file1.pdf, page 5; file2.txt").

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
logger.debug("RAG prompt template created.")


def format_docs_for_prompt(docs: list) -> dict:
    if not docs:
        return {"context": "No context provided.", "citations": ["N/A"]}

    formatted_lines = []
    citations = set()
    for i, doc in enumerate(docs):
        citation = doc.metadata.get('citation', 'Unknown Source')
        citations.add(citation)
        formatted_lines.append(f"--- Context Chunk {i+1} (Citation: {citation}) ---\n{doc.page_content}")

    full_context = "\n\n".join(formatted_lines)
    sorted_citations = sorted(list(citations))
    logger.debug(f"Formatted {len(docs)} chunks for prompt. Citations: {sorted_citations}")
    return {"context": full_context, "citations": sorted_citations}


def get_rag_chain(retriever, llm):
    if not retriever or not llm:
        logger.error("Cannot build RAG chain: Retriever or LLM is missing.")
        return None

    logger.info(f"Building RAG chain with Retriever: {retriever.__class__.__name__} (k={retriever.search_kwargs.get('k', 'N/A')}), LLM: {llm.__class__.__name__} (Temp={getattr(llm, 'temperature', 'N/A')})")

    setup_and_retrieval = RunnableParallel(
        retrieved_info=RunnableLambda(lambda question: retriever.invoke(question)) | RunnableLambda(format_docs_for_prompt),
        question=RunnablePassthrough()
    )
    prepare_prompt_input = RunnableLambda(
        lambda x: {
            "context": x['retrieved_info']['context'],
            "question": x['question']
        }
    )
    rag_chain = (
        setup_and_retrieval
        | prepare_prompt_input
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain constructed successfully.")
    return rag_chain


def get_retrieved_context_for_display(retriever, question: str) -> tuple[list | None, str]:
    if not retriever:
        logger.warning("Retriever not available for get_retrieved_context_for_display.")
        return None, "Retriever not available."
    if not question:
        return [], "No question provided."

    logger.debug(f"Retrieving context for display for question: '{question[:50]}...' using retriever with k={retriever.search_kwargs.get('k', 'N/A')}")
    try:
        retrieved_docs = retriever.invoke(question)
        formatted_info = format_docs_for_prompt(retrieved_docs)
        logger.debug(f"Retrieved {len(retrieved_docs)} docs for display.")
        return retrieved_docs, formatted_info["context"]
    except Exception as e:
        logger.error(f"Error retrieving context for display: {e}", exc_info=True)
        return None, f"Error retrieving context: {e}"

logger.info("RAG pipeline components defined.")