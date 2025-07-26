🦙 Groq-Powered RAG Chatbot using LLaMA3
A lightning-fast Retrieval-Augmented Generation (RAG) chatbot powered by Groq's LLaMA3 70B model with ultra-low latency LPUs (Language Processing Units). This application allows you to chat with your documents using state-of-the-art AI technology.

My Motivation
I built this project to gain hands-on experience with Retrieval-Augmented Generation (RAG) systems and to explore the incredible speed of Groq's LPU inference engine. My goal was to create a practical tool that I could use to quickly query and understand my own documents, like research papers or technical manuals.

✨ Features
⚡ Ultra-Fast Inference: Powered by Groq's Language Processing Units (LPUs) for real-time responses

🦙 LLaMA3 70B Model: Uses Meta's advanced open-source language model

📄 Document Upload: Support for PDF and TXT files

🧠 Conversational Memory: Maintains context across multiple turns

📚 Source Citations: Automatically cites source documents in responses

🎨 Modern UI: Clean and intuitive Streamlit interface

🔒 Secure: API keys are handled securely

🚀 Quick Start
1. Prerequisites
Python 3.9 or higher

Groq API key (free at console.groq.com)

2. Installation
Bash

# Clone or download this project
# Navigate to the project directory
# Install dependencies
pip install -r requirements.txt
3. Setup API Key
Get your free Groq API key:

Visit Groq Console

Sign up/Login

Create a new API key

Copy the API key

4. Run the Application
Bash

streamlit run main.py
The application will open in your browser at http://localhost:8501

🎯 How to Use
Enter API Key: Paste your Groq API key in the sidebar

Upload Documents: Upload PDF or TXT files you want to chat with

Process Documents: Click "Process Documents" to create the knowledge base

Start Chatting: Ask questions about your documents!

🏗️ Architecture
User Input → Document Processing → Vector Store (FAISS) → LangChain RAG Chain → Groq LLaMA3 → Response with Citations
Key Components:
Groq LLaMA3 70B: Ultra-fast language model for generation

FAISS: Facebook's vector database for similarity search

LangChain: Framework for chaining LLM operations

Sentence Transformers: For document embeddings

Streamlit: Web interface framework

📋 Dependencies
This project's dependencies are listed in the requirements.txt file.

streamlit
langchain
langchain-groq
faiss-cpu
sentence-transformers
pypdf
python-dotenv
groq

🔧 Configuration
Model Settings
Model: llama3-70b-8192

Temperature: 0.1 (for more deterministic responses)

Max Tokens: 1024

Chunk Size: 750 tokens

Chunk Overlap: 150 tokens

Supported File Types
PDF files (.pdf)

Text files (.txt)

🛠️ Customization
Modify Model Parameters
Edit the setup_groq_llm() function in main.py:

Python

llm = ChatGroq(
    temperature=0.1,  # Adjust for creativity vs consistency
    model_name="llama3-70b-8192",  # Or use other Groq models like 'llama3-8b-8192'
    max_tokens=1024,  # Adjust response length
)
Change Chunk Settings
Modify the create_vectorstore() function:

Python

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,    # Increase for longer context, decrease for more specific answers
    chunk_overlap=150,  # Adjust overlap
)
🚨 Troubleshooting
Common Issues
"Error setting up Groq LLM"

Check your API key is correct.

Ensure you have an internet connection.

Verify API key permissions on the Groq Console.

"No documents were processed successfully"

Check file formats (PDF/TXT only).

Ensure files aren't corrupted.

Try smaller files first.

🔒 Security Notes
API keys are handled in Streamlit's session state and are not stored permanently.

Uploaded files are processed in temporary directories and deleted after use.

Use environment variables or Streamlit Secrets for production deployments.

🙏 Acknowledgments
Groq for providing ultra-fast LLM inference

Meta for the LLaMA models

LangChain for the RAG framework

Streamlit for the web interface

Facebook Research for FAISS

Built with ❤️ for learning and experimentation with cutting-edge AI technology!