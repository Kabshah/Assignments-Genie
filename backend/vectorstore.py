import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings # Changed to HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

# Import API keys from config (only Pinecone is needed here now)
from config import PINECONE_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
INDEX_NAME="rag-index"

# Track the most recent document uploaded
_last_uploaded_document = None

def clear_index():
    """Delete all data from the Pinecone index and reset tracking."""
    global _last_uploaded_document
    try:
        if INDEX_NAME in pc.list_indexes().names():
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True)
            print(f"Cleared all data from index '{INDEX_NAME}'")
        _last_uploaded_document = None
    except Exception as e:
        print(f"Error clearing index: {e}")

def get_retriever():
    # ensure index exists,create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index")
        pc.create_index(
            name=INDEX_NAME,
            metric="cosine",
            dimension=384,
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
        )
        print("Index created sucessfully")

    # Always return the retriever, whether index was just created or already existed
    vectorstore=PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})
    

def add_document(text_content:str, document_name:str = "uploaded_document"):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    Includes document metadata to track which PDF each chunk came from.
    """
    global _last_uploaded_document
    
    if not text_content:
     raise ValueError("Document content can't be processed because its empty")
    
    text_splitter=RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=200,
       add_start_index=True,
       separators=["\n\n", "\n", ". ", " ", ""]
    )
    #create langchain document objects from the raw text
    documents=text_splitter.create_documents([text_content])

    # Add metadata to each document chunk to track which PDF it came from
    for doc in documents:
        doc.metadata = {
            "source": document_name,
            "chunk_index": documents.index(doc)
        }

    print(f"Splitting '{document_name}' into chunks for indexing...")
    print(f"Created {len(documents)} chunks from document")

    # get vectorstore instance to add documents
    vectorstore=PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings)

    #add documents to vectorstore
    vectorstore.add_documents(documents)
    print(f"Successfully indexed {len(documents)} chunks from '{document_name}'")
    
    # Track the most recently uploaded document
    _last_uploaded_document = document_name
    print("Sucessfully added chunks to vector store")