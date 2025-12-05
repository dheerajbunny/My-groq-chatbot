# Configuration file for Groq-Powered RAG Chatbot
# Modify these settings to customize the chatbot behavior

# Model Configuration
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.1
MAX_TOKENS = 1024

# Alternative models available on Groq:
# - "llama-3.1-8b-instant" (faster)
# - "mixtral-8x7b-32768" (alternative)
# - "gemma-7b-it" (lightweight)

# Document Processing
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
MAX_FILE_SIZE_MB = 10

# Retrieval Settings
SIMILARITY_SEARCH_K = 3  # Number of chunks to retrieve
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Alternative embedding models:
# - "all-mpnet-base-v2" (better quality)
# - "all-roberta-large-v1" (highest quality)
# - "multi-qa-MiniLM-L6-cos-v1" (Q&A optimized)

# Memory Configuration
MEMORY_TYPE = "buffer"  # Options: buffer, window, summary, summary_buffer
MEMORY_WINDOW_SIZE = 5  # For window memory type

# UI Configuration
PAGE_TITLE = "Groq-Powered RAG Chatbot"
PAGE_ICON = "🦙"
LAYOUT = "wide"  # Options: wide, centered

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "txt"]

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """You are an AI assistant powered by Groq's LLaMA3 model. Use the following pieces of context to answer the user's question accurately and comprehensively.

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

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = "chatbot.log"

# Performance Settings
ENABLE_CACHING = True
CACHE_TTL = 3600  # Cache time-to-live in seconds

# Security Settings
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = ["pdf", "txt"]

# Development Settings
DEBUG_MODE = False
VERBOSE_LOGGING = False
