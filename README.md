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
