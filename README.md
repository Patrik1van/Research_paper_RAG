# Local Research Paper RAG 📚

A local, Dockerized Retrieval-Augmented Generation (RAG) application to chat with your research papers. 

This project uses a decoupled architecture:
* **Frontend:** [Open WebUI](https://docs.openwebui.com/) (A sleek, ChatGPT-like interface)
* **Backend:** FastAPI
* **Agent Framework:** LangGraph & LangChain
* **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash` & `text-embedding-004`)
* **Vector Database:** FAISS (In-memory)

## 📂 Project Structure

Make sure your repository looks exactly like this before running:

```text
Research_paper_RAG/
├── docker-compose.yml
├── README.md
├── .env                  <-- You must create this
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── app.py
    └── data/             <-- Put your PDF files here
```

## ⚙️ Prerequisites

1. **Docker Desktop** installed and running on your machine.
2. A **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/)).

## 🚀 Getting Started

### 1. Set up your Environment Variables
In the root directory (next to `docker-compose.yml`), create a file named `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Add your Research Papers
Create the `backend/data/` folder if it doesn't exist, and drop at least one `.pdf` file inside it. 
*(Note: If the folder is empty on startup, the search tool will be initialized without any documents).*

### 3. Build and Run the Application
Open your terminal in the root directory and run:
```bash
docker compose up --build
```
Wait a minute or two for the containers to build, download, and embed your initial PDFs.

## 💬 Usage

1. **Open the UI:** Go to [http://localhost:3000](http://localhost:3000) in your web browser.
2. **Sign Up:** Create an account (this is completely local; the first account automatically becomes the Admin).
3. **Select the Agent:** At the top of the chat interface, click the model dropdown and select **`langgraph-rag-agent`**.
4. **Chat:** Ask questions about your research paper! For best results, explicitly ask the agent to search (e.g., *"Use your search tool to find the conclusion of the paper."*)

## 🔄 Updating Documents on the Fly

If you add a new PDF to the `backend/data/` folder while the server is running, the agent won't automatically know about it. 

To force the backend to rescan the folder and embed the new documents, open a new terminal window and run:
```bash
curl -X POST http://localhost:8000/v1/refresh
```
You will see the backend logs confirm that the new documents have been loaded and embedded.
```
