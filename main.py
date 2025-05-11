import os
import requests
from sympy import sympify, SympifyError
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
import streamlit as st

# Use Streamlit secrets for Hugging Face API token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise ValueError("Hugging Face API token not found in Streamlit secrets!")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

def load_documents(docs_folder="docs"):
    documents = []
    try:
        for filename in os.listdir(docs_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(docs_folder, filename)
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        return documents
    except Exception as e:
        raise ValueError(f"Error loading documents: {e}")

def create_vectorstore(documents, embedding_model, index_path="faiss_index"):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        vectorstore.save_local(index_path)
        return vectorstore

def rag_qa(query, llm, vectorstore):
    prompt_template = """You are a knowledge assistant. Answer the question based solely on the provided context. If the context does not contain relevant information, state: "I don't have specific information about this topic based on the provided documents." Do not use external knowledge.

    Context: {context}

    Question: {question}

    Answer: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = PROMPT.format(context=context, question=query)
    try:
        answer = llm.invoke(prompt).strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"
    return {"answer": answer, "context": context}

def calculator(query):
    try:
        expr = query.lower().replace("calculate", "").replace("add", "").strip()
        result = sympify(expr).evalf()
        return str(result)
    except SympifyError:
        return "Invalid mathematical expression."
    except Exception as e:
        return f"Calculation error: {e}"

def define_word(query):
    word = query.lower().replace("define", "").strip()
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            meaning = data[0]["meanings"][0]["definitions"][0]["definition"]
            return f"'{word}': {meaning}"
        return f"No definition found for '{word}'."
    except requests.RequestException:
        return "Error fetching definition."

def initialize_agent_and_tools(llm, vectorstore):
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Use for mathematical calculations with BODMAS (e.g., 'calculate (2 + 3) * 4')"
        ),
        Tool(
            name="Define Word",
            func=define_word,
            description="Use for defining words via Free Dictionary API (e.g., 'define apple')"
        ),
        Tool(
            name="RAG QA Pipeline",
            func=lambda query: rag_qa(query, llm, vectorstore)["answer"],
            description="Use for answering questions based on document retrieval"
        )
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=False
    )
    return agent, tools

def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = load_documents()
    if not documents:
        print("No documents found in 'docs' folder.")
        return
    vectorstore = create_vectorstore(documents, embedding_model)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        max_new_tokens=200,
        temperature=0.7
    )
    agent, _ = initialize_agent_and_tools(llm, vectorstore)
    print("Type a question or 'exit' to quit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        try:
            response = agent.invoke(user_input)
            tool_used = "RAG QA Pipeline"
            context_snippets = ""
            if "calculate" in user_input.lower() or "add" in user_input.lower():
                tool_used = "Calculator"
            elif "define" in user_input.lower():
                tool_used = "Define Word"
            else:
                result = rag_qa(user_input, llm, vectorstore)
                context_snippets = result["context"]
            print(f"\nTool Used: {tool_used}")
            if context_snippets:
                print(f"Context Snippets:\n{context_snippets}\n")
            print(f"Answer: {response['output']}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
