"""
Demo script for testing the Groq-Powered RAG Chatbot components
This script demonstrates how to use the chatbot programmatically
"""

import os
import tempfile
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def test_groq_connection(api_key):
    """Test connection to Groq API"""
    try:
        llm = ChatGroq(
            temperature=0.1,
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            max_tokens=100
        )

        # Test with simple prompt
        response = llm.invoke("Hello! Please respond with 'Connection successful!'")
        print(f"✅ Groq connection test: {response.content}")
        return llm
    except Exception as e:
        print(f"❌ Groq connection failed: {str(e)}")
        return None

def create_sample_document():
    """Create a sample document for testing"""
    sample_text = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence. 
    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
    that can learn and improve from experience without being explicitly programmed.

    Large Language Models (LLMs) are a type of AI model that can understand and generate 
    human-like text. They are trained on vast amounts of text data and can perform various 
    natural language processing tasks such as text generation, translation, summarization, 
    and question answering.

    Retrieval-Augmented Generation (RAG) is a technique that combines the power of LLMs 
    with external knowledge sources. Instead of relying solely on the model's training data, 
    RAG systems can retrieve relevant information from a knowledge base and use it to 
    generate more accurate and up-to-date responses.

    Groq is a company that develops specialized AI inference hardware called Language 
    Processing Units (LPUs). These chips are designed specifically for running language 
    models and can provide significantly faster inference speeds compared to traditional 
    GPUs, making real-time AI applications more feasible.
    """

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(sample_text)
        tmp_file_path = tmp_file.name

    return tmp_file_path, sample_text

def test_document_processing():
    """Test document loading and processing"""
    print("📄 Testing document processing...")

    # Create sample document
    tmp_file_path, original_text = create_sample_document()

    try:
        # Load document
        loader = TextLoader(tmp_file_path)
        documents = loader.load()

        # Add metadata
        for doc in documents:
            doc.metadata['source'] = 'sample_ai_document.txt'
            doc.metadata['upload_time'] = datetime.now().isoformat()

        print(f"✅ Loaded {len(documents)} document(s)")
        print(f"   Content length: {len(documents[0].page_content)} characters")

        return documents

    except Exception as e:
        print(f"❌ Document processing failed: {str(e)}")
        return None
    finally:
        # Clean up
        os.unlink(tmp_file_path)

def test_embeddings_and_vectorstore(documents):
    """Test embeddings and vector store creation"""
    print("🧠 Testing embeddings and vector store...")

    try:
        # Setup embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for demo
            chunk_overlap=100,
            length_function=len,
        )

        splits = text_splitter.split_documents(documents)
        print(f"✅ Created {len(splits)} document chunks")

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        print("✅ Vector store created successfully")

        # Test similarity search
        query = "What is RAG?"
        results = vectorstore.similarity_search(query, k=2)
        print(f"✅ Similarity search returned {len(results)} results for: '{query}'")

        return vectorstore

    except Exception as e:
        print(f"❌ Embeddings/Vector store test failed: {str(e)}")
        return None

def test_rag_chain(llm, vectorstore):
    """Test the complete RAG chain"""
    print("🔗 Testing RAG chain...")

    try:
        # Custom prompt template
        prompt_template = """You are an AI assistant. Use the following context to answer the question accurately.

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

        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False
        )

        print("✅ RAG chain created successfully")

        # Test questions
        test_questions = [
            "What is RAG?",
            "How does Groq help with AI applications?",
            "What's the difference between AI and ML?",
            "Can you summarize what we discussed about LLMs?"
        ]

        print("\n🤖 Testing conversation flow:")
        print("-" * 50)

        for i, question in enumerate(test_questions, 1):
            print(f"\nQ{i}: {question}")

            try:
                response = conversation_chain({"question": question})
                answer = response["answer"]
                sources = response.get("source_documents", [])

                print(f"A{i}: {answer}")

                if sources:
                    print(f"   📚 Sources: {len(sources)} document chunk(s)")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")

        print("-" * 50)
        print("✅ RAG chain testing completed")

        return conversation_chain

    except Exception as e:
        print(f"❌ RAG chain test failed: {str(e)}")
        return None

def main():
    """Main demo function"""
    print("🦙 Groq-Powered RAG Chatbot Demo")
    print("=" * 50)

    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()

    if not api_key:
        print("❌ No API key provided. Exiting.")
        return

    # Test components step by step
    print("\n1. Testing Groq connection...")
    llm = test_groq_connection(api_key)
    if not llm:
        return

    print("\n2. Testing document processing...")
    documents = test_document_processing()
    if not documents:
        return

    print("\n3. Testing embeddings and vector store...")
    vectorstore = test_embeddings_and_vectorstore(documents)
    if not vectorstore:
        return

    print("\n4. Testing RAG chain...")
    conversation_chain = test_rag_chain(llm, vectorstore)
    if not conversation_chain:
        return

    print("\n🎉 All tests completed successfully!")
    print("\nThe chatbot is ready to use. Run 'streamlit run main.py' to start the web interface.")

if __name__ == "__main__":
    main()
