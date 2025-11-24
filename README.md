# Dr. Kokoro – RAG Chatbot (LangChain + Vertex AI + Streamlit)

This project is a Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **Google Gemini (Vertex AI)**, **ChromaDB**, and **Streamlit**.  
It loads PDF documents, turns them into vector embeddings, retrieves context, and answers user questions with high accuracy.  
It also includes a **human takeover mode**, custom UI, and persistent conversation state.

---

##  Features
- RAG pipeline using **MiniLM-L6-v2** embeddings  
- **Google Gemini 1.5 Flash** (Vertex AI) as the LLM  
- Elegant **Streamlit chat interface** with custom CSS  
- **ChromaDB** for vector search & context retrieval  
- **Human takeover trigger** (“human help”, “contact support”)  
- Refresh knowledge base button  
- Shared state stored in `shared_state.json`  

---

##  Tech Stack
- **LangChain**
- **Google Vertex AI**
- **Streamlit UI**
- **HuggingFace Embeddings**
- **Chroma Vector Database**
- **PyPDFLoader**
- **FileLock**

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt


Tech Stack
Layer	Technology
LLM	Google Gemini (Vertex AI)
Framework	LangChain
Embeddings	HuggingFace MiniLM-L6-v2
Vector DB	Chroma
Frontend	Streamlit
PDF Loader	PyPDFLoader
Memory	JSON state + FileLock
 Project Structure
 dr-kokoro-chatbot
│
├── app.py                 # Your main Streamlit app (one file project)
├── shared_state.json      # For human takeover + welcome message
├── customer chatbot/            
└── README.md

How to Run
1️ Install dependencies
pip install -r requirements.txt

2️ Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="key.json"

3️ Update your GCP project ID inside app.py
project="YOUR_GCP_PROJECT_ID"

4️ Run Streamlit
streamlit run app.py

⚙️ What the Chatbot Does
🔹 1. Load/Embed PDF

Splits PDF → Embeds using MiniLM → Stores in ChromaDB.

🔹 2. Retrieve Relevant Chunks

Queries are matched with top-k chunks (k=3).

🔹 3. Pass Context to Gemini Model

Prompt includes:

Use the PDF context and respond elegantly.
If you lack context, say: "My deepest apologies..."

🔹 4. Human Takeover

If user types:

human help
contact support


Admin manually responds.

📜 Example Usage

User:

"Explain the condition mentioned on Page 3."

Bot (with RAG):

Summarizes from PDF context.

User:

"I need human help."

Bot:

“Aditya is assisting you live.”

📄 Requirements
streamlit
langchain
langchain-community
langchain-google-vertexai
langchain-huggingface
langchain-chroma
chromadb
pypdf
google-cloud-aiplatform
filelock
uuid

👤 Author

Aditya Guleria
RAG Engineering • LangChain • Vertex AI
