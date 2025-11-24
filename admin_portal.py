import streamlit as st
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
from filelock import FileLock
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure shared_state.json exists
if not os.path.exists("shared_state.json"):
    with FileLock("shared_state.json.lock"):
        with open("shared_state.json", "w") as f:
            json.dump({
                "welcome_message": "Welcome to Kokoro Doctor, a premier AI-powered heart health service. Enquire about our services, pricing, or support.",
                "human_takeover": False,
                "latest_query": {},
                "latest_response": {},
                "queries": {},
                "admin_name": "Aditya"
            }, f)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f0e6f0 0%, #d9b9d9 100%); font-family: 'Georgia', serif; color: #2c1a3d; padding: 2rem; }
    .admin-section { background: #ffffff; padding: 2rem; border-radius: 1rem; box-shadow: 0 6px 12px rgba(0,0,0,0.2); margin-bottom: 2rem; }
    .welcome-display { background: #f9f3ff; padding: 1.5rem; border-left: 5px solid #6a0dad; border-radius: 0.75rem; margin-bottom: 2rem; }
    .stButton>button { background: #6a0dad; color: white; border-radius: 0.5rem; padding: 0.75rem 2rem; border: none; }
    .stButton>button:hover { background: #5a009d; }
    .query-item { background: #f9f3ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 5px solid #6a0dad; }
    .status-pending { color: #ff9800; }
    .status-pending-human { color: #2196f3; }
    .status-in-progress { color: #2196f3; }
    .status-resolved { color: #4caf50; }
    .status-customer-confirmed { color: #388e3c; }
    </style>
    """, unsafe_allow_html=True)

# Process PDFs
def process_pdfs(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for uploaded_file in uploaded_files:
        try:
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(f"temp_{uploaded_file.name}")
            documents.extend(loader.load())
            os.remove(f"temp_{uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
            logger.error(f"PDF processing error: {str(e)}")
            continue
    if documents:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
        st.success("PDFs processed and embedded into the knowledge base.")
        logger.info("PDFs processed successfully.")
    else:
        st.warning("No valid PDFs processed.")

# Save shared state
def save_shared_state():
    with FileLock("shared_state.json.lock"):
        with open("shared_state.json", "w") as f:
            state = {
                "welcome_message": st.session_state.get("welcome_message", ""),
                "human_takeover": st.session_state.get("human_takeover", False),
                "latest_query": st.session_state.get("latest_query", {}),
                "latest_response": st.session_state.get("latest_response", {}),
                "queries": st.session_state.get("queries", {}),
                "admin_name": st.session_state.get("admin_name", "Aditya")
            }
            json.dump(state, f, indent=4)
    logger.info("Shared state saved.")

# Main Admin Interface
def main():
    load_css()
    st.sidebar.title("Dr. Kokoro Admin Portal")

    # Load shared state
    if "queries" not in st.session_state:
        with FileLock("shared_state.json.lock"):
            try:
                with open("shared_state.json", "r") as f:
                    state = json.load(f)
                    st.session_state.queries = state.get("queries", {})
                    st.session_state.welcome_message = state.get("welcome_message", "")
                    st.session_state.latest_query = state.get("latest_query", {})
                    st.session_state.latest_response = state.get("latest_response", {})
                    st.session_state.admin_name = state.get("admin_name", "Aditya")
                    st.session_state.human_takeover = state.get("human_takeover", False)
            except FileNotFoundError:
                st.session_state.queries = {}
                st.session_state.welcome_message = ""
                st.session_state.latest_query = {}
                st.session_state.latest_response = {}
                st.session_state.admin_name = "Aditya"
                st.session_state.human_takeover = False

    # Display welcome message
    with st.container():
        st.markdown('<div class="welcome-display">Current Welcome Message:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="welcome-display">{st.session_state.welcome_message}</div>', unsafe_allow_html=True)

    # Welcome Message Editor
    with st.sidebar.container():
        st.markdown('<div class="admin-section">', unsafe_allow_html=True)
        st.subheader("Edit Welcome Message")
        new_welcome = st.text_area("Compose Welcome Message", st.session_state.welcome_message, height=100)
        if st.button("Update Welcome", key="update_welcome"):
            st.session_state.welcome_message = new_welcome
            save_shared_state()
            st.success("Welcome message updated!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Admin Name Configuration
    with st.sidebar.container():
        st.markdown('<div class="admin-section">', unsafe_allow_html=True)
        st.subheader("Set Admin Name")
        new_admin_name = st.text_input("Enter Admin Name", st.session_state.admin_name)
        if st.button("Update Name", key="update_admin_name"):
            st.session_state.admin_name = new_admin_name
            save_shared_state()
            st.success("Admin name updated!")
        st.markdown('</div>', unsafe_allow_html=True)

    # PDF Upload
    with st.sidebar.container():
        st.markdown('<div class="admin-section">', unsafe_allow_html=True)
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="pdf_uploader")
        if uploaded_files and st.button("Process PDFs", key="process_pdfs"):
            with st.spinner("Processing documents..."):
                process_pdfs(uploaded_files)
        st.markdown('</div>', unsafe_allow_html=True)

    # Query Management
    with st.sidebar.container():
        st.markdown('<div class="admin-section">', unsafe_allow_html=True)
        st.subheader("Query Management")
        if st.session_state.queries:
            for query_id, query_data in st.session_state.queries.items():
                with st.expander(f"Query: {query_data['content']}"):
                    st.write(f"**Timestamp**: {query_data['timestamp']}")
                    status = query_data.get("status", "Pending")
                    st.write(f"**Status**: <span class='status-{status.lower().replace(' ', '-')}'>{status}</span>", unsafe_allow_html=True)
                    if status in ["Pending", "Pending Human", "In Progress"]:
                        response = st.text_area("Craft Response", key=f"response_{query_id}", height=100, value=query_data.get("response", ""))
                        if st.button("Take Over", key=f"takeover_{query_id}"):
                            with FileLock("shared_state.json.lock"):
                                with open("shared_state.json", "r+") as f:
                                    state = json.load(f)
                                    state["queries"][query_id]["status"] = "In Progress"
                                    state["queries"][query_id]["assigned_to"] = st.session_state.admin_name
                                    state["human_takeover"] = True
                                    f.seek(0)
                                    json.dump(state, f, indent=4)
                            st.rerun()
                        if st.button("Mark Resolved", key=f"resolve_{query_id}"):
                            if response:
                                with FileLock("shared_state.json.lock"):
                                    with open("shared_state.json", "r+") as f:
                                        state = json.load(f)
                                        state["queries"][query_id]["status"] = "Resolved"
                                        state["queries"][query_id]["response"] = response
                                        state["queries"][query_id]["last_updated"] = str(datetime.now())
                                        state["queries"][query_id]["assigned_to"] = st.session_state.admin_name
                                        state["latest_response"] = {"query_id": query_id, "content": response, "timestamp": str(datetime.now())}
                                        state["human_takeover"] = False
                                        f.seek(0)
                                        json.dump(state, f, indent=4)
                                st.rerun()
                            else:
                                st.warning("Provide a response before resolving.")
                    elif status in ["Resolved", "Customer Confirmed"]:
                        st.write(f"**Response**: {query_data.get('response', 'N/A')}")
                        st.write(f"**Resolved At**: {query_data.get('last_updated', 'N/A')}")
        else:
            st.write("No queries available.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
