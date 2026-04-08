import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT","us-east-1") # e.g., "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index") # using this as the main index name

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Embedding Model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DOC_SOURCE_DIR=os.getenv("DOC_SOURCE_DIR","data")