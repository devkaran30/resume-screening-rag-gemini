
import sys
sys.dont_write_bytecode = True

import os
import time

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Gemini OpenAI-compatible API endpoint
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
DEFAULT_MODEL = "gemini-3.8-flash"

st.set_page_config(page_title="Resume Screening — Gemini")
st.title("Resume Screening — Gemini")

# --------------------------------------------------
# Application messages
# --------------------------------------------------

welcome_message = """
#### Introduction 🚀

This RAG-based resume screening system helps hiring managers
find relevant candidates from a collection of resumes.

It retrieves relevant resumes and uses an LLM to help with
candidate analysis, summarization, and job-description matching.

#### Getting started 🛠️

1. Enter your Gemini API key in the sidebar.
2. Select a Gemini model.
3. Enter a job description or ask a question about candidates.

The default knowledge base uses synthetic resumes.

You can find example data in the project's
[synthetic resumes dataset](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv)
and example job descriptions in the
[job description dataset](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/supplementary-data/job_title_des.csv).
"""

info_message = """
# Information

### 1. How can I use my own resumes?

Upload a CSV file using the sidebar. It must contain these columns:

- `Resume`
- `ID`

Indexing may take some time, depending on the dataset size.

### 2. Can I change the model or RAG mode?

Yes. Select the Gemini model and RAG mode from the sidebar.

The temperature is set to `0.1` in the LLM configuration.
The retriever uses the existing project configuration.

### 3. Is my uploaded data safe?

The application does not intentionally save uploaded data as part
of the Streamlit session. However, resume text included in prompts
may be sent to the configured Gemini API for processing.

Use synthetic or appropriately authorized data when testing.

### 4. How does the chatbot work?

The application retrieves relevant resumes and passes them to the
LLM for answering questions. The retriever also determines how
RAG should be applied to a query.
"""

about_message = """
# About

This is a prototype for exploring RAG-based resume screening.

Original project:
[Resume Screening RAG Pipeline on GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline)
"""


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=welcome_message)
    ]

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "rag_selection" not in st.session_state:
    st.session_state.rag_selection = "Generic RAG"

if "gpt_selection" not in st.session_state:
    st.session_state.gpt_selection = DEFAULT_MODEL

if "df" not in st.session_state:
    if not DATA_PATH or not os.path.exists(DATA_PATH):
        st.error(
            "Dataset not found. Check DATA_PATH in your .env file."
        )
        st.stop()

    st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

if "rag_pipeline" not in st.session_state:
    if not FAISS_PATH or not os.path.exists(FAISS_PATH):
        st.error(
            "FAISS vectorstore not found. Check FAISS_PATH in your .env file."
        )
        st.stop()

    vectordb = FAISS.load_local(
        FAISS_PATH,
        st.session_state.embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True,
    )

    st.session_state.rag_pipeline = SelfQueryRetriever(
        vectordb,
        st.session_state.df,
    )


# --------------------------------------------------
# Upload resumes
# --------------------------------------------------

def upload_file():
    uploaded_file = st.session_state.get("uploaded_file")

    if uploaded_file is None:
        # Restore the default synthetic-resume dataset.
        st.session_state.df = pd.read_csv(DATA_PATH)

        vectordb = FAISS.load_local(
            FAISS_PATH,
            st.session_state.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )

        st.session_state.rag_pipeline = SelfQueryRetriever(
            vectordb,
            st.session_state.df,
        )
        st.session_state.resume_list = []
        return

    modal = Modal(
        key="resume_upload_error",
        title="File Error",
        max_width=500,
    )

    try:
        df_load = pd.read_csv(uploaded_file)

        if "Resume" not in df_load.columns or "ID" not in df_load.columns:
            with modal.container():
                st.error(
                    'The CSV must contain both "Resume" and "ID" columns.'
                )
            return

        if df_load.empty:
            with modal.container():
                st.error("The uploaded CSV contains no rows.")
            return

        if df_load["Resume"].isna().all():
            with modal.container():
                st.error("The Resume column does not contain any resume text.")
            return

        with st.spinner("Indexing uploaded resumes. Please wait..."):
            st.session_state.df = df_load

            vectordb = ingest(
                st.session_state.df,
                "Resume",
                st.session_state.embedding_model,
            )

            # Use the same state variable that the chat flow reads.
            st.session_state.rag_pipeline = SelfQueryRetriever(
                vectordb,
                st.session_state.df,
            )

            st.session_state.resume_list = []

        st.success("Resumes indexed successfully.")

    except Exception as error:
        with modal.container():
            st.error("Unable to process the uploaded CSV.")
            st.code(str(error))


# --------------------------------------------------
# Gemini API key/model validation
# --------------------------------------------------

def check_gemini_api_key(api_key: str, model_name: str):
    """
    Test the key and model through Gemini's OpenAI-compatible endpoint.

    Returns:
        (True, message) when the test succeeds.
        (False, error_message) when it fails.
    """
    if not api_key or not api_key.strip():
        return False, "Please enter your Gemini API key."

    if not model_name or not model_name.strip():
        return False, "Please enter a Gemini model ID."

    url = f"{GEMINI_BASE_URL}chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name.strip(),
        "messages": [
            {"role": "user", "content": "Reply with OK."}
        ],
        "max_tokens": 5,
        "temperature": 0,
    }

    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=30.0,
        )

        if response.status_code == 200:
            return True, "Gemini API connection successful."

        # Do not display or log the API key.
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get(
                "message", response.text
            )
        except ValueError:
            error_message = response.text

        return (
            False,
            f"Gemini API returned HTTP {response.status_code}: "
            f"{error_message}",
        )

    except httpx.HTTPError as error:
        return False, f"Could not connect to Gemini API: {error}"


# --------------------------------------------------
# Chat controls
# --------------------------------------------------

def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [
        AIMessage(content=welcome_message)
    ]


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("# Control Panel")

    st.text_input(
        "Gemini API Key",
        type="password",
        key="api_key",
        help="Get your key from Google AI Studio. Do not share it.",
    )

    st.selectbox(
        "RAG Mode",
        ["Generic RAG", "RAG Fusion"],
        key="rag_selection",
    )

    st.text_input(
        "Gemini Model ID",
        value=DEFAULT_MODEL,
        key="gpt_selection",
        help="Enter a model ID supported by your Gemini API key.",
    )

    st.file_uploader(
        "Upload resumes (CSV)",
        type=["csv"],
        key="uploaded_file",
        on_change=upload_file,
    )

    st.button(
        "Clear conversation",
        on_click=clear_message,
    )

    st.divider()
    st.markdown(info_message)

    st.divider()
    st.markdown(about_message)
    st.markdown("Original project by [Hungreeee](https://github.com/Hungreeee)")


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    else:
        # Preserve the project's retriever-verbosity display.
        with st.chat_message("assistant"):
            message[0].render(*message[1:])


# --------------------------------------------------
# Validate Gemini API credentials
# --------------------------------------------------

api_key = st.session_state.api_key.strip()
model_name = st.session_state.gpt_selection.strip()

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to continue.")
    st.stop()

# Cache the validation result for the current key/model combination.
# Changing the key or model automatically triggers a new validation.
validation_pair = (api_key, model_name)

if st.session_state.get("_gemini_validation_pair") != validation_pair:
    with st.spinner("Checking Gemini API key and model..."):
        is_valid, validation_message = check_gemini_api_key(
            api_key,
            model_name,
        )

    st.session_state["_gemini_validation_pair"] = validation_pair
    st.session_state["_gemini_validation_result"] = (
        is_valid,
        validation_message,
    )

is_valid, validation_message = st.session_state[
    "_gemini_validation_result"
]

if not is_valid:
    st.error(validation_message)
    st.stop()


# --------------------------------------------------
# Initialize chatbot and handle the user query
# --------------------------------------------------

retriever = st.session_state.rag_pipeline

llm = ChatBot(
    api_key=api_key,
    model=model_name,
)

user_query = st.chat_input("Type your message here...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append(
        HumanMessage(content=user_query)
    )

    with st.chat_message("assistant"):
        start = time.time()

        with st.spinner("Retrieving resumes and generating an answer..."):
            document_list = retriever.retrieve_docs(
                user_query,
                llm,
                st.session_state.rag_selection,
            )

            query_type = retriever.meta_data["query_type"]
            st.session_state.resume_list = document_list

            # Keep the original project's history argument unchanged.
            stream_message = llm.generate_message_stream(
                user_query,
                document_list,
                [],
                query_type,
            )

            response = st.write_stream(stream_message)

        elapsed_time = time.time() - start

        retriever_message = chatbot_verbosity
        retriever_message.render(
            document_list,
            retriever.meta_data,
            elapsed_time,
        )

        st.session_state.chat_history.append(
            AIMessage(content=response)
        )

        st.session_state.chat_history.append(
            (
                retriever_message,
                document_list,
                retriever.meta_data,
                elapsed_time,
            )
        )
