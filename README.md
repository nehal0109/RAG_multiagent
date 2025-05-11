RAG-Powered Multi-Agent Q&A
Overview
This project implements a Retrieval-Augmented Generation (RAG) powered multi-agent Q&A system as per the assignment requirements. It ingests text documents, indexes them in a FAISS vector store, and uses an agent to route queries to appropriate tools (calculator, dictionary, or RAG pipeline) powered by a Hugging Face LLM. It supports both a minimal CLI and a Streamlit web UI.
Architecture

Data Ingestion:
Loads .txt files from the docs folder (e.g., company FAQs, product specs).
Chunks documents using CharacterTextSplitter (chunk size: 500, overlap: 50).


Vector Store & Retrieval:
Uses FAISS with sentence-transformers/all-MiniLM-L6-v2 embeddings.
Retrieves top 3 relevant chunks for RAG queries.


LLM Integration:
Uses mistralai/Mixtral-8x7B-Instruct-v0.1 via HuggingFaceEndpoint for robust response generation.


Agentic Workflow:
Implements an agent using LangChain’s AgentExecutor that routes queries:
To Calculator for "calculate" or "add" queries (e.g., "calculate (2 + 3) * 4") using sympy for BODMAS.
To Define Word for "define" queries (e.g., "define apple") using the Free Dictionary API.
To RAG QA Pipeline for other queries (e.g., "What is the return policy?").


Logs decisions via verbose output (optional in CLI).


Demo Interface:
CLI: Minimal interface showing tool used, context snippets (for RAG), and answer.
Web UI: Streamlit interface with input box and formatted output for tool, context snippets, and answer.



Key Design Choices

AgentExecutor: Used for compatibility with HuggingFaceEndpoint, despite deprecation.
Mixtral LLM: Chosen for strong instruction-following.
Free Dictionary API: Provides real-time word definitions.
Sympy: Ensures BODMAS-compliant calculations offline.
FAISS: Simple, local storage for small document collections.
Streamlit: Lightweight web UI for interactive demo.

Prerequisites

Python 3.8+
A Hugging Face API token (free tier works for Mixtral)
A docs folder with 3–5 .txt files (e.g., FAQs, product specs)
Internet access for Free Dictionary API and Hugging Face API

Setup

Clone the Repository (or create files manually):git clone <your-repo-url>
cd rag_multiagent_assistant


Create a Virtual Environment:python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows PowerShell


Install Dependencies:pip install langchain langchain-community langchain-huggingface faiss-cpu python-dotenv sympy requests streamlit


Set Up Environment Variables:Create a .env file in the project root:HUGGINGFACEHUB_API_TOKEN=your_token_here


Prepare Documents:Create a docs folder and add 3–5 .txt files with content (e.g., FAQs, product specs).
Run the Application:
CLI:python main.py


Web UI:streamlit run app.py





Usage

CLI:
Run python main.py.
Type questions (e.g., calculate (2 + 3) * 4, define apple, What is the return policy?).
View tool used, context snippets (for RAG), and answer.
Type exit to quit.


Web UI:
Run streamlit run app.py.
Enter questions in the input box.
Click "Submit" to see tool used, context snippets (for RAG), and answer.
Click "Clear" to reset the input.



Example Interaction (Web UI)

Enter: calculate (2 + 3) * 4
Tool Used: Calculator
Answer: 20


Enter: define apple
Tool Used: Define Word
Answer: 'apple': The round fruit of a tree of the rose family...


Enter: What is the return policy?
Tool Used: RAG QA Pipeline
Context Snippets: [Relevant document excerpts...]
Answer: The return policy allows returns within 30 days with a receipt.



Limitations

Mixtral requires a valid Hugging Face API token and internet access.
Free Dictionary API may have rate limits; errors are handled gracefully.
FAISS is local and may not scale for large document collections.
AgentExecutor is deprecated; future versions could use LangGraph with a compatible LLM.

Future Improvements

Migrate to LangGraph with a compatible chat model.
Enhance Streamlit UI with styling or history tracking.
Support additional tools (e.g., web search, date/time queries).
Use a more robust embedding model (e.g., all-mpnet-base-v2) for better RAG performance.

