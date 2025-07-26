# Author: Pavan Dheeraj

import streamlit as st
import os
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tempfile
import uuid

# Page configuration
st.set_page_config(
    page_title="Groq-Powered RAG Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #E3F2FD;
    border-left: 4px solid #2196F3;
}
.assistant-message {
    background-color: #F3E5F5;
    border-left: 4px solid #9C27B0;
}
.citation {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

# Setup Groq LLM
@st.cache_resource
def setup_groq_llm(api_key):
    try:
        llm = ChatGroq(
            temperature=0.1,
            groq_api_key=api_key,
            model_name="llama3-70b-8192",
            max_tokens=1024
        )
        return llm
    except Exception as e:
        st.error(f"Error setting up Groq LLM: {str(e)}")
        return None

# Setup embeddings
@st.cache_resource
def setup_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Document processing
def process_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load document based on file type
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue

            docs = loader.load()

            # Add metadata
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['upload_time'] = datetime.now().isoformat()

            documents.extend(docs)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    return documents

# Create vector store
def create_vectorstore(documents, embeddings):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    splits = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore

# Setup conversation chain
def setup_conversation_chain(llm, vectorstore):
    # Custom prompt template with citation instructions
    prompt_template = """You are an AI assistant powered by Groq's LLaMA3 model. Use the following pieces of context to answer the user's question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
1. Base your answer primarily on the provided context
2. Always include citations by mentioning the source document name
3. If information is not in the context, clearly state that
4. Be conversational and helpful
5. Provide detailed answers when possible

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Create retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

# Display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                st.markdown("**Sources:**")
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

# Format response with citations
def format_response_with_citations(response, source_documents):
    answer = response["answer"]
    citations = []

    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown')
        if source not in citations:
            citations.append(source)

    return answer, citations

# Main application
def main():
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🦙 Groq-Powered RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### A personal project to chat with documents using Groq and LLaMA3")

    ## Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # Check for secrets and load API key automatically
        if 'GROQ_API_KEY' in st.secrets:
            st.success("API key found in Secrets!", icon="✅")
            st.session_state.groq_api_key = st.secrets['GROQ_API_KEY']
        else:
            # Fallback to manual input if secret is not found
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.groq_api_key,
                help="Get your API key from https://console.groq.com"
            )
            if groq_api_key:
                st.session_state.groq_api_key = groq_api_key

        st.divider()

        # Document upload
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to chat with"
        )

        if uploaded_files and st.session_state.groq_api_key:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Process documents
                    documents = process_documents(uploaded_files)

                    if documents:
                        # Setup embeddings and create vector store
                        embeddings = setup_embeddings()
                        vectorstore = create_vectorstore(documents, embeddings)
                        st.session_state.vectorstore = vectorstore

                        # Setup LLM and conversation chain
                        llm = setup_groq_llm(st.session_state.groq_api_key)
                        if llm:
                            conversation_chain = setup_conversation_chain(llm, vectorstore)
                            st.session_state.conversation_chain = conversation_chain

                            st.success(f"Successfully processed {len(documents)} document chunks!")
                    else:
                        st.error("No documents were processed successfully.")

        # Session info
        st.divider()
        st.header("ℹ️ Session Info")
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        st.write(f"**Messages:** {len(st.session_state.messages)}")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # Main chat interface
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to get started.")
        st.markdown("""
        **To get started:**
        1. Get your free API key from [Groq Console](https://console.groq.com)
        2. Enter your API key in the sidebar
        3. Upload some documents
        4. Start chatting!
        """)
        return

    if not st.session_state.conversation_chain:
        st.info("📄 Please upload and process some documents to start chatting.")
        return

    # Display chat messages
    display_chat_messages()

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation_chain({"question": prompt})
                    answer, citations = format_response_with_citations(
                        response, 
                        response.get("source_documents", [])
                    )

                    st.markdown(answer)

                    if citations:
                        st.markdown("**Sources:**")
                        for citation in citations:
                            st.markdown(f"- {citation}")

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "citations": []
                    })

if __name__ == "__main__":
    main()
