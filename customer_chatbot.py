import streamlit as st
import os
import json
import uuid
import logging
from datetime import datetime
from filelock import FileLock

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Modern import for Vertex AI
from langchain_google_vertexai import ChatVertexAI

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Google Gemini (Vertex AI) Chat Model ---
def init_google_flan_model():
    try:
        # You can change to "gemini-1.5-pro" for a stronger model
        llm = ChatVertexAI(
            model="gemini-1.5-flash",  # Newer Google model (replaces chat-bison)
            temperature=0,
            max_output_tokens=1024,
            project="YOUR_GCP_PROJECT_ID",   # 👈 replace with your GCP project ID
            location="us-central1"           # 👈 your Vertex AI location
        )
        return llm
    except Exception as e:
        st.warning(f"⚠️ Google Vertex AI initialization failed: {e}")
        logger.error(f"Vertex AI init error: {str(e)}")
        return None

flan_client = init_google_flan_model()

# --- Streamlit Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "welcome_message" not in st.session_state:
    st.session_state.welcome_message = "Welcome to Dr. Kokoro, your distinguished AI heart health companion. 💙"
if "human_takeover" not in st.session_state:
    st.session_state.human_takeover = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "human_response" not in st.session_state:
    st.session_state.human_response = None
if "admin_name" not in st.session_state:
    st.session_state.admin_name = "Aditya"

# --- Shared State Loader ---
def load_shared_state():
    with FileLock("shared_state.json.lock"):
        try:
            if os.path.exists("shared_state.json"):
                with open("shared_state.json", "r") as f:
                    content = f.read()
                    state = json.loads(content) if content.strip() else {}
                st.session_state.human_takeover = state.get("human_takeover", False)
                st.session_state.welcome_message = state.get("welcome_message", st.session_state.welcome_message)
                st.session_state.human_response = state.get("latest_response", None)
                st.session_state.admin_name = state.get("admin_name", "Aditya")
        except Exception as e:
            logger.error(f"Error loading shared state: {str(e)}")

# --- CSS ---
def load_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5eef5, #d9b9d9); font-family: 'Georgia', serif; color: #2c1a3d; }
    .chat-container { max-width: 900px; margin: 0 auto; border: 3px solid #8e44ad; border-radius: 20px;
        background: rgba(255,255,255,0.95); box-shadow: 0 10px 20px rgba(0,0,0,0.2); padding: 2rem; }
    .chat-message { padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 5px 10px rgba(0,0,0,0.1);
        font-size: 1.1rem; line-height: 1.8; width: fit-content; max-width: 80%; }
    .chat-message.user { background: linear-gradient(90deg, #e6e6fa, #d1c4e9); margin-left: auto; }
    .chat-message.bot { background: linear-gradient(90deg, #f9f3ff, #e1bee7); margin-right: auto; }
    .chat-message.human { background: linear-gradient(90deg, #c8e6c9, #a5d6a7); font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- RAG Initialization ---
def initialize_rag():
    if not os.path.exists("./chroma_db"):
        st.warning("⚠️ No Chroma database found. Please upload a PDF from admin panel.")
        return
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        prompt_template = """Use the PDF context and respond elegantly. 
If you lack context, say: 'My deepest apologies, I am unable to provide insight at this time.'

{context}
Enquiry: {question}
Response:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=flan_client if flan_client else None,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        st.success("✅ RAG pipeline initialized successfully.")
    except Exception as e:
        st.error(f"RAG initialization failed: {str(e)}")
        logger.error(str(e))

# --- Main Chat ---
def main():
    load_css()
    load_shared_state()
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.title("💬 Dr. Kokoro Customer Chatbot")

    if st.button("🔄 Refresh Knowledge Base"):
        initialize_rag()

    # Display chat history
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "bot", "content": st.session_state.welcome_message})

    for msg in st.session_state.messages:
        role = msg["role"]
        st.markdown(f'<div class="chat-message {role}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Please enter your enquiry...")
    if prompt:
        query_id = str(uuid.uuid4())
        st.session_state.messages.append({"role": "user", "content": prompt, "query_id": query_id})

        if any(x in prompt.lower() for x in ["human help", "contact support"]):
            st.session_state.messages.append({"role": "human", "content": f"{st.session_state.admin_name} is assisting you live."})
            st.session_state.human_takeover = True
            st.rerun()
        else:
            try:
                if st.session_state.qa_chain:
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                else:
                    if flan_client:
                        response = flan_client.invoke([HumanMessage(content=prompt)]).content
                    else:
                        response = "⚠️ AI service unavailable. Please try again later."
                st.session_state.messages.append({"role": "bot", "content": response})
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                st.session_state.messages.append({
                    "role": "bot",
                    "content": "My deepest apologies, I am unable to provide insight at this time."
                })

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    initialize_rag()
    main()
