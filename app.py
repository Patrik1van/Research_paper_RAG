import os
import time
from typing import List, Optional, Annotated, Sequence, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# LangChain & LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. FastAPI Setup & Pydantic Models (OpenAI Format) ---
app = FastAPI(title="LangGraph RAG Backend")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0
    stream: Optional[bool] = False

# Global variables to store our graph and tool
retriever_tool = None
graph = None

# --- NEW: Reusable Graph Builder Function ---
def build_agent_graph(tools):
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent") 

    return workflow.compile()


# --- 2. RAG Setup (Runs on Startup) ---
@app.on_event("startup")
async def startup_event():
    global retriever_tool, graph
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY is not set!")
        return

    print("Loading documents from /app/data...")
    loader = PyPDFDirectoryLoader("/app/data")
    docs = loader.load()
    
    if not docs:
        print("No PDFs found in /app/data. Tool will be empty.")
        docs = [SystemMessage(content="No documents loaded.")]
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Embedding {len(splits)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    retriever_tool = create_retriever_tool(
        retriever,
        "research_paper_search",
        "Searches and returns excerpts from the loaded research papers. Use this to answer questions about the documents."
    )
    
    # Build the graph initially
    graph = build_agent_graph([retriever_tool])
    print("LangGraph Agent initialized successfully!")


# --- 3. Refresh Endpoint (Call this when you add a PDF) ---
@app.post("/v1/refresh")
async def refresh_documents():
    global retriever_tool, graph
    
    print("Rescanning documents from /app/data...")
    loader = PyPDFDirectoryLoader("/app/data")
    docs = loader.load()
    
    if not docs:
        return {"status": "error", "message": "No PDFs found."}
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Re-embedding {len(splits)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    retriever_tool = create_retriever_tool(
        retriever,
        "research_paper_search",
        "Searches and returns excerpts from the loaded research papers."
    )
    
    # Recompile the graph with the brand new tool
    graph = build_agent_graph([retriever_tool])
    
    return {"status": "success", "message": f"Successfully reloaded and embedded {len(docs)} documents."}


# --- 4. Open WebUI Required Endpoints ---
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "langgraph-rag-agent", 
                "object": "model", 
                "created": int(time.time()), 
                "owned_by": "custom backend"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized. Check API key and data.")

    lc_messages = [
        SystemMessage(content=(
            "You are an expert research assistant. You have access to a loaded research paper via the `research_paper_search` tool. "
            "ALWAYS use the tool to search the document before answering. "
            "If the user asks for a summary, search for keywords like 'Abstract', 'Introduction', or 'Conclusion' to gather context."
        ))
    ]
    
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(SystemMessage(content=msg.content)) 

    inputs = {"messages": lc_messages}
    
    try:
        final_state = graph.invoke(inputs)
        raw_content = final_state["messages"][-1].content
        
        # --- FIX FOR [object Object] ---
        # Gemini sometimes returns content as a list of dictionaries instead of a plain string.
        if isinstance(raw_content, list):
            final_message = ""
            for block in raw_content:
                if isinstance(block, dict) and "text" in block:
                    final_message += block["text"]
                elif isinstance(block, str):
                    final_message += block
        else:
            final_message = str(raw_content)
            
    except Exception as e:
        print(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    response_data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_message
            },
            "finish_reason": "stop"
        }]
    }
    
    return response_data