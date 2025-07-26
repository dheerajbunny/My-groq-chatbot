#!/bin/bash

echo "🦙 Setting up Groq-Powered RAG Chatbot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "📝 Next steps:"
echo "1. Get your Groq API key from https://console.groq.com"
echo "2. Run: streamlit run main.py"
echo "3. Enter your API key in the sidebar"
echo "4. Upload documents and start chatting!"
