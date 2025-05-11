import streamlit as st
from main import load_documents, create_vectorstore, rag_qa, calculator, define_word, initialize_agent_and_tools
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

st.set_page_config(page_title="RAG-Powered Q&A", layout="wide")

if "agent" not in st.session_state:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = load_documents()
    if not documents:
        st.error("No documents found in 'docs' folder.")
        st.stop()
    vectorstore = create_vectorstore(documents, embedding_model)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        max_new_tokens=200,
        temperature=0.7
    )
    agent, tools = initialize_agent_and_tools(llm, vectorstore)
    st.session_state.agent = agent
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = llm
    st.session_state.tools = tools

st.title("RAG-Powered Multi-Agent Q&A")
st.write("Enter a question to get an answer from the Calculator, Define Word, or RAG QA Pipeline.")

with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("Your Question:",
                             placeholder="e.g., calculate (2 + 3) * 4, define apple, or What is the return policy?")
    submit_button = st.form_submit_button("Submit")
    clear_button = st.form_submit_button("Clear")

if submit_button and question:
    try:
        agent = st.session_state.agent
        vectorstore = st.session_state.vectorstore
        llm = st.session_state.llm
        response = agent.invoke(question)
        tool_used = "RAG QA Pipeline"
        context_snippets = ""
        if "calculate" in question.lower() or "add" in question.lower():
            tool_used = "Calculator"
        elif "define" in question.lower():
            tool_used = "Define Word"
        else:
            result = rag_qa(question, llm, vectorstore)
            context_snippets = result["context"]
        st.subheader("Results")
        st.write(f"**Tool Used**: {tool_used}")
        if context_snippets:
            st.write("**Context Snippets**:")
            st.text(context_snippets)
        st.write("**Answer**:")
        st.write(response["output"])
    except Exception as e:
        st.error(f"Error: {e}")

if clear_button:
    st.experimental_rerun()
