# src/ui/app.py

import streamlit as st
import os
import logging
import sys
import time
from copy import deepcopy

try:
    # Assuming optional centralized logging setup
    from src.utils.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

try:
    from src import config
except ImportError:
    st.error("FATAL: Cannot import configuration from src.config.")
    logger.critical("FATAL: Cannot import configuration from src.config.")
    st.stop()

try:
    from src.pipeline.rag import (
        load_embedding_function, load_retriever, load_llm,
        get_rag_chain, get_retrieved_context_for_display, format_docs_for_prompt
    )
except ImportError as e:
     st.error(f"FATAL: Failed to import RAG pipeline components: {e}.")
     logger.critical(f"FATAL: Failed to import RAG pipeline components: {e}.", exc_info=True)
     st.stop()

st.set_page_config(
    page_title="AI Tutor Pro",
    page_icon="🧑‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSISTANT_AVATAR = "🎓"
USER_AVATAR = "🧑‍🎓"

# Initialize session state
if "selected_llm_provider_name" not in st.session_state:
    st.session_state.selected_llm_provider_name = config.DEFAULT_LLM_PROVIDER_NAME
if "current_retriever_k" not in st.session_state:
    st.session_state.current_retriever_k = config.RETRIEVER_K
if "current_llm_temperature" not in st.session_state:
    default_llm_config = config.AVAILABLE_LLMS.get(st.session_state.selected_llm_provider_name or "", {})
    st.session_state.current_llm_temperature = default_llm_config.get('temperature', 0.5)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Load Resources Sequentially based on current session state ---
# Assumes load_* functions in rag.py use @st.cache_resource correctly
embedding_function = load_embedding_function()
embedding_status = embedding_function is not None

current_retriever = None
retriever_status = False
if embedding_status:
    try:
        current_retriever = load_retriever(embedding_function, k=st.session_state.current_retriever_k)
        retriever_status = current_retriever is not None
    except Exception as e:
        logger.error(f"Failed to load retriever in main flow: {e}", exc_info=True)

current_llm = None
llm_status = False
if st.session_state.selected_llm_provider_name and config.AVAILABLE_PROVIDERS:
    try:
        current_llm = load_llm(
            provider_name=st.session_state.selected_llm_provider_name,
            temperature=st.session_state.current_llm_temperature
        )
        llm_status = current_llm is not None
    except Exception as e:
        logger.error(f"Failed to load LLM in main flow: {e}", exc_info=True)

# --- Build/Rebuild RAG Chain and store in session state ---
# Determine if a rebuild is needed based on component status changes or if chain is missing
# Compare current loaded components with potentially stored ones (or lack thereof)
chain_needs_update = False
if 'loaded_retriever_k' not in st.session_state or st.session_state.loaded_retriever_k != st.session_state.current_retriever_k:
    chain_needs_update = True
if 'loaded_llm_provider' not in st.session_state or st.session_state.loaded_llm_provider != st.session_state.selected_llm_provider_name:
    chain_needs_update = True
if 'loaded_llm_temp' not in st.session_state or st.session_state.loaded_llm_temp != st.session_state.current_llm_temperature:
    chain_needs_update = True
if st.session_state.rag_chain is None:
    chain_needs_update = True
if not retriever_status or not llm_status: # Force rebuild if components failed
    chain_needs_update = True
    st.session_state.rag_chain = None # Clear chain if components failed

chain_status = False
if retriever_status and llm_status:
    if chain_needs_update:
        logger.info(f"Rebuilding RAG chain: k={st.session_state.current_retriever_k}, provider={st.session_state.selected_llm_provider_name}, temp={st.session_state.current_llm_temperature}")
        try:
            st.session_state.rag_chain = get_rag_chain(current_retriever, current_llm)
            if st.session_state.rag_chain:
                # Store parameters used for this build
                st.session_state.loaded_retriever_k = st.session_state.current_retriever_k
                st.session_state.loaded_llm_provider = st.session_state.selected_llm_provider_name
                st.session_state.loaded_llm_temp = st.session_state.current_llm_temperature
                logger.info("RAG chain rebuild successful.")
            else:
                 logger.error("get_rag_chain returned None during rebuild.")
                 st.session_state.rag_chain = None # Ensure it's None
        except Exception as e:
            logger.error(f"Error rebuilding RAG chain: {e}", exc_info=True)
            st.session_state.rag_chain = None
            st.error(f"Error building RAG processing pipeline: {e}")

# Final chain status check based on session state object
chain_status = st.session_state.rag_chain is not None

# --- Sidebar ---
with st.sidebar:
    st.header("🧑‍🏫 AI Tutor Pro Config")
    st.divider()
    st.subheader("System Status")
    embed_status_display = "✅ Ready" if embedding_status else "❌ Error"
    retriever_status_display = "✅ Ready" if retriever_status else "⏳ Error/Pending"
    llm_status_display = "✅ Ready" if llm_status else ("⚪ N/A" if not st.session_state.selected_llm_provider_name else "⏳ Error/Pending")
    chain_status_display = "✅ Ready" if chain_status else "⏳ Error/Pending"
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Embeddings:** {embed_status_display}")
        st.markdown(f"**Knowledge Base:** {retriever_status_display}")
    with col2:
        st.markdown(f"**Language Model:** {llm_status_display}")
        st.markdown(f"**RAG Chain:** {chain_status_display}")
    st.divider()

    st.subheader("Language Model")
    if not config.AVAILABLE_PROVIDERS:
         st.error("No LLM providers available. Check API keys in `.env` and restart.")
    else:
        try:
            # Handle potential None value during initial runs or if config is bad
            current_selection = st.session_state.selected_llm_provider_name
            if current_selection not in config.AVAILABLE_PROVIDERS and config.AVAILABLE_PROVIDERS:
                logger.warning(f"Invalid LLM selection '{current_selection}' detected, resetting to default.")
                current_selection = config.AVAILABLE_PROVIDERS[0]
                st.session_state.selected_llm_provider_name = current_selection # Update state immediately
            current_selection_index = config.AVAILABLE_PROVIDERS.index(current_selection) if current_selection else 0
        except (ValueError, TypeError):
             current_selection_index = 0
             if config.AVAILABLE_PROVIDERS:
                 st.session_state.selected_llm_provider_name = config.AVAILABLE_PROVIDERS[0]

        st.selectbox(
            "Select Model:",
            options=config.AVAILABLE_PROVIDERS,
            index=current_selection_index,
            key="selected_llm_provider_name",
            help="Choose the AI model."
        )

        st.slider(
            "Temperature (Creativity):",
            min_value=0.0, max_value=1.0,
            key="current_llm_temperature",
            step=0.05,
            help="Adjust creativity."
        )

    st.divider()
    st.subheader("Knowledge Retrieval")
    st.number_input(
        "Context Chunks (k):",
        min_value=1, max_value=10,
        key="current_retriever_k",
        step=1,
        help="Number of text chunks for context."
    )

    st.divider()
    st.subheader("📚 Indexed Materials")
    try:
        source_file_path = config.SOURCE_INFO_FILE
        if os.path.exists(source_file_path):
            with open(source_file_path, 'r', encoding='utf-8') as f:
                source_files = sorted([line.strip() for line in f if line.strip()])
            if source_files:
                with st.expander("View Indexed Sources", expanded=False):
                    for file in source_files: st.markdown(f"- `{file}`")
            else: st.caption("Source list empty.")
        else: st.caption("Source list file not found.")
    except Exception as e:
        logger.error(f"Error reading source info file {config.SOURCE_INFO_FILE}: {e}")
        st.error("Could not load source list.")

    st.divider()
    if st.button("Clear Chat History", key="clear_chat", use_container_width=True):
        logger.info("Clear chat button clicked.")
        st.session_state.chat_history = []
        st.rerun()

# --- Main Chat Interface ---
st.title("🧑‍🏫 AI Tutor Pro")
active_llm_name = st.session_state.selected_llm_provider_name or "N/A"
st.markdown(f"Ask questions about your course materials. Using: **{active_llm_name}**")
st.divider()

for message in st.session_state.chat_history:
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources Consulted", expanded=False):
                st.markdown(message["sources"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    if not chain_status or st.session_state.rag_chain is None:
        logger.warning("RAG chain not ready when processing user query.")
        err_msg = "Sorry, the AI Tutor is not ready. Please check the sidebar status, ensure API keys are correct in `.env`, and potentially restart the application."
        st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": ""})
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.error(err_msg)
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.status(f"Consulting knowledge base with {st.session_state.selected_llm_provider_name}...", expanded=False) as status:
                assistant_response_content = ""
                sources_md = "*No specific sources retrieved or an error occurred.*"
                try:
                    st.write("🧠 Thinking...")
                    start_time = time.time()

                    assistant_response_content = st.session_state.rag_chain.invoke(prompt)

                    end_time = time.time()
                    logger.info(f"RAG chain invocation took {end_time - start_time:.2f} seconds.")
                    status.update(label="📚 Retrieving sources...", state="running", expanded=False)

                    # Use the retriever loaded earlier in this run
                    retrieved_docs, _ = get_retrieved_context_for_display(current_retriever, prompt)

                    if retrieved_docs:
                        formatted_info = format_docs_for_prompt(retrieved_docs)
                        citations = formatted_info.get("citations", [])
                        if citations:
                            sources_list_html = "<ul>" + "".join(f"<li><code>{cit}</code></li>" for cit in citations) + "</ul>"
                            sources_md = sources_list_html
                        else:
                             sources_md = "*Retrieved context, but citation metadata missing.*"
                    else:
                        sources_md = "*No relevant context sections found.*"

                    status.update(label=f"Answer generated by {st.session_state.selected_llm_provider_name}.", state="complete", expanded=False)

                except Exception as e:
                    logger.exception("Error processing user query:")
                    error_details = f"{type(e).__name__}" + (f": {e}" if str(e) else "")
                    assistant_response_content = f"⚠️ Sorry, an error occurred: {error_details}"
                    sources_md = f"*Error during processing: {error_details}*"
                    status.update(label="Processing failed", state="error", expanded=True)
                    st.error(assistant_response_content)

            st.markdown(assistant_response_content)
            with st.expander("🔍 View Sources Consulted", expanded=False):
                st.markdown(sources_md, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response_content,
            "sources": sources_md
        })

if not st.session_state.chat_history:
     st.info("Welcome! Ask a question using the input below.")