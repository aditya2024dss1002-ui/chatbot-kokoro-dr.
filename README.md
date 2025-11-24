Dr. Kokoro – RAG Chatbot (LangChain +  Streamlit)

Dr. Kokoro is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, Google Gemini (Vertex AI), ChromaDB, and Streamlit.
It supports PDF knowledge ingestion, elegant UI, context-aware answering, and human agent takeover.

 Features

1. RAG Pipeline (PDF → Embeddings → ChromaDB → Context Retrieval)

2. Google Gemini 1.5 Flash (via Vertex AI)

3. Streamlit Chat UI with custom gradient design

4. Human Takeover Mode triggered by keywords

5. Persistent shared state (shared_state.json)

6. Beautiful chat bubbles (User / Bot / Human)

7. Refresh Knowledge Base button

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

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="key.json"

3️⃣ Update your GCP project ID inside app.py
project="YOUR_GCP_PROJECT_ID"

4️⃣ Run Streamlit
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
