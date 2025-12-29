# ====================== Enhanced RAG Pipeline with Docling Integration ======================
# Version: 2.0
# Purpose: This is a Retrieval-Augmented Generation (RAG) system that:
#   1. Loads PDF documents and converts them to searchable text chunks
#   2. Creates a vector database (FAISS) to find relevant information
#   3. Uses an AI model (Ollama) to answer questions based on the documents
#   4. Can work in three modes: hybrid (RAG + AI), RAG-only, or AI-only

# ====================== Imports (External Libraries) ====================== 
import os              # For file system operations (paths, directories)
import sys             # For system-specific parameters
import glob            # For finding files matching patterns (*.pdf)
import json            # For saving/loading JSON metadata
import requests        # For making HTTP requests to Ollama API
import torch           # PyTorch library for checking GPU availability
import subprocess      # For running system commands (nvidia-smi)
import logging         # For logging messages to console and file
import threading       # For thread locks (prevent concurrent rebuilds)
from pathlib import Path
from typing import List, Dict, Optional, Any  # For type hints
from datetime import datetime                  # For timestamps

# FastAPI framework for creating REST API endpoints
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field  # For API request/response validation
from dotenv import load_dotenv         # For loading environment variables from .env file

# Docling: Advanced PDF document processing library
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# LangChain: Framework for building LLM applications
from langchain_huggingface import HuggingFaceEmbeddings  # Converts text to vector embeddings
from langchain_community.vectorstores import FAISS       # Vector database for similarity search
from langchain_core.documents import Document            # Document object structure
from sentence_transformers import CrossEncoder           # Re-ranks search results for better accuracy

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Optional: FastEmbed - Faster embedding generation (if available)
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("⚠️  FastEmbed not available. Using HuggingFace embeddings.")

# ====================== Logging Configuration ======================
# Set up logging to track what the system is doing
# Logs are written to both: 1) 'rag_pipeline.log' file, 2) console output
logging.basicConfig(
    level=logging.INFO,  # Show INFO level and above (INFO, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Include timestamp, name, level, message
    handlers=[
        logging.FileHandler('rag_pipeline.log'),  # Save to file
        logging.StreamHandler()                   # Print to console
    ]
)
logger = logging.getLogger(__name__)

# Suppress transformer tokenization warnings (reduces console clutter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== Configuration Management ======================
# Load environment variables from .env file (if exists)
load_dotenv()

class RAGConfig:
    """
    Centralized configuration class that stores all system settings.
    Values are loaded from environment variables with sensible defaults.
    """
    # Path to folder containing PDF documents to process
    FOLDER_PATH = os.getenv("FOLDER_PATH", "./data/")
    
    # Path where the vector database (FAISS index) is saved/loaded
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    
    # Name of the Ollama AI model to use for generating answers
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:7b")
    
    # URL endpoint for the local Ollama API
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/completions")
    
    # Chunking configuration: How to split documents into smaller pieces
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 300))  # Maximum tokens per chunk (optimal for embeddings)
    CHUNK_TOKENIZER = os.getenv("CHUNK_TOKENIZER", "sentence-transformers/all-MiniLM-L12-v2")
    
    # Retrieval configuration: How many document chunks to retrieve
    TOP_K = int(os.getenv("TOP_K", 8))              # Initial number of chunks to retrieve
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 3))  # Final number after re-ranking
    
    # Embedding configuration: Which model converts text to vectors
    USE_FASTEMBED = os.getenv("USE_FASTEMBED", "false").lower() == "true" and FASTEMBED_AVAILABLE
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5" if USE_FASTEMBED else "sentence-transformers/all-MiniLM-L12-v2")
    
    # Device configuration: GPU/CPU settings
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # Force CPU for RTX 5090 compatibility
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")        # Force CPU for RTX 5090 compatibility
    USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"  # Disable reranker by default
    
    # Security configuration
    API_KEY = os.getenv("API_KEY", None)  # API key for authentication (None = no auth in dev mode)
    
    # LLM configuration: How the AI model generates responses
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", 2500))  # Max length of generated answer
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))  # Lower = more focused, higher = more creative

# Create a global config instance
config = RAGConfig()

# Global variables that will be initialized later:
vectorstore = None       # FAISS vector database - stores document embeddings
retriever = None         # LangChain retriever - searches the vector database
reranker = None          # Cross-encoder model - re-ranks search results
embedding_model = None   # Model that converts text to vectors
document_metadata = {}   # Tracks which documents have been loaded and when

# Rebuild lock to prevent concurrent index rebuilds
_rebuild_lock = threading.Lock()
_is_rebuilding = False

# ====================== GPU Diagnostics ======================
def get_gpu_info():
    """
    Check and display GPU information for hardware acceleration.
    GPU speeds up embedding generation and model inference significantly.
    If no GPU is found, the system will use CPU (slower but still works).
    """
    logger.info("=" * 50)
    logger.info("🔍 Checking GPU Information...")
    logger.info("=" * 50)

    # Check if CUDA (NVIDIA GPU support) is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"✅ CUDA available. Number of GPUs: {num_gpus}")

        # List all available GPUs
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU {i}: {gpu_name}")

        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")

        # Try to get NVIDIA driver version using nvidia-smi command
        try:
            driver_info = subprocess.check_output(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader", 
                shell=True
            )
            logger.info(f"NVIDIA Driver: {driver_info.decode().strip()}")
        except Exception:
            logger.warning("Could not get NVIDIA driver version")
    else:
        logger.warning("❌ No GPU detected. Running on CPU.")

    logger.info("=" * 50)

# Run GPU diagnostics when the script loads
get_gpu_info()

# ====================== Enhanced Document Loading with Docling ======================
def load_pdfs_with_docling(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load and convert PDF files using Docling library.
    
    Docling is better than basic PDF readers because it:
    - Understands document structure (headings, tables, lists)
    - Preserves formatting and layout context
    - Handles complex PDFs with images and multi-column text
    
    Args:
        folder_path: Path to folder containing PDF files (e.g., "./data/")
        
    Returns:
        List of dictionaries, each containing a Docling document and its metadata
        Returns empty list if no PDFs found (system will run in LLM-only mode)
    """
    # Find all PDF files in the specified folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in: {folder_path}")
        logger.info("System will continue in LLM-only mode without RAG capabilities.")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # ========== CHANGE THIS SECTION (around line 178-180) ==========
    # OLD CODE (remove this):
    # converter = DocumentConverter()
    
    # NEW CODE (add this):
    from docling.datamodel.base_models import InputFormat
    
    # Initialize Docling's document converter with device configuration
    converter = DocumentConverter()
    # ========== END OF CHANGES ==========
    
    all_documents = []
    
    # Process each PDF file
    for pdf_path in pdf_files:
        try:
            logger.info(f"📄 Loading with Docling: {pdf_path}")
            
            # Convert PDF to Docling document object
            result = converter.convert(source=pdf_path)
            doc = result.document
            
            # Create metadata dictionary to track document information
            doc_metadata = {
                'source': pdf_path,                          # Full path to PDF file
                'filename': os.path.basename(pdf_path),       # Just the filename
                'loaded_at': datetime.now().isoformat(),      # When it was loaded
                'num_pages': getattr(doc, 'num_pages', 'unknown'),  # Number of pages (if available)
            }
            
            # Add document and metadata to our collection
            all_documents.append({
                'document': doc,
                'metadata': doc_metadata
            })
            
            logger.info(f"✅ Loaded: {os.path.basename(pdf_path)}")
            
        except Exception as e:
            # If a PDF fails to load, log the error and continue with other files
            logger.error(f"❌ Error loading {pdf_path}: {e}", exc_info=True)
            continue
    
    if all_documents:
        logger.info(f"✅ Successfully loaded {len(all_documents)} documents with Docling")
    else:
        logger.warning("⚠️  No documents could be loaded. System will run in LLM-only mode.")
    
    return all_documents

# ====================== Hybrid Semantic Chunking ======================
def chunk_documents_hybrid(documents: List[Dict[str, Any]]) -> List[Document]:
    """
    Split large documents into smaller, meaningful chunks.
    
    Why chunking is needed:
    - Embedding models have token limits (typically 512 tokens)
    - Smaller chunks = more precise retrieval of relevant information
    - Semantic chunking keeps related sentences together
    
    How HybridChunker works:
    - Respects document structure (doesn't split mid-paragraph)
    - Merges related sections (merge_peers=True)
    - Adds surrounding context to each chunk for better understanding
    
    Args:
        documents: List of document dictionaries from load_pdfs_with_docling
        
    Returns:
        List of LangChain Document objects (each is a chunk with metadata)
    """
    logger.info("🔧 Initializing HybridChunker for semantic chunking...")
    
    try:
        # Initialize the chunker with configuration
        chunker = HybridChunker(
            tokenizer=config.CHUNK_TOKENIZER,  # Model used to count tokens
            max_tokens=config.MAX_TOKENS,      # Maximum tokens per chunk (300)
            merge_peers=True                   # Merge related sections for better context
        )
    except Exception as e:
        # If HybridChunker fails, use simpler fallback method
        logger.error(f"Failed to initialize HybridChunker: {e}")
        logger.info("Falling back to basic chunking")
        return fallback_chunking(documents)
    
    all_chunks = []
    
    # Process each document
    for doc_data in documents:
        try:
            doc = doc_data['document']
            metadata = doc_data['metadata']
            
            # Split document into chunks using HybridChunker
            chunks = list(chunker.chunk(dl_doc=doc))
            logger.info(f"Created {len(chunks)} chunks from {metadata['filename']}")
            
            # Convert each chunk to LangChain Document format
            for i, chunk in enumerate(chunks):
                # Try to add surrounding context to the chunk (enriched_text)
                try:
                    enriched_text = chunker.contextualize(chunk=chunk)
                except:
                    # If contextualization fails, just use the chunk text
                    enriched_text = chunk.text
                
                # Create metadata for this specific chunk
                chunk_metadata = {
                    'source': metadata['source'],        # Original PDF file path
                    'filename': metadata['filename'],    # PDF filename
                    'chunk_id': i,                       # Chunk number (0, 1, 2, ...)
                    'total_chunks': len(chunks),         # How many chunks from this document
                    'loaded_at': metadata['loaded_at']   # When document was loaded
                }
                
                # Create LangChain Document object (combines text + metadata)
                langchain_doc = Document(
                    page_content=enriched_text,    # The actual text content
                    metadata=chunk_metadata        # Information about the chunk
                )
                all_chunks.append(langchain_doc)
                
        except Exception as e:
            # If chunking fails for a document, log error and continue
            logger.error(f"Error chunking {doc_data['metadata']['filename']}: {e}", exc_info=True)
            continue
    
    logger.info(f"✅ Created {len(all_chunks)} semantically coherent chunks total")
    return all_chunks

def fallback_chunking(documents: List[Dict[str, Any]]) -> List[Document]:
    """
    Simpler chunking method used when HybridChunker fails.
    
    Uses RecursiveCharacterTextSplitter which:
    - Splits text by trying different separators in order (paragraphs, lines, sentences, words)
    - Maintains some overlap between chunks for context continuity
    - Less sophisticated than HybridChunker but more reliable
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        List of LangChain Document objects with chunks
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    logger.info("Using fallback RecursiveCharacterTextSplitter")
    
    # Initialize text splitter with configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,           # Target size for each chunk (in characters)
        chunk_overlap=300,         # How much to overlap between chunks (for context)
        separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
    )
    
    all_chunks = []
    for doc_data in documents:
        try:
            # Convert Docling document to plain text string
            doc_text = str(doc_data['document'])
            
            # Split text into chunks
            chunks = text_splitter.create_documents(
                [doc_text],                    # Text to split
                metadatas=[doc_data['metadata']]  # Metadata to attach to each chunk
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Fallback chunking error: {e}")
            continue
    
    return all_chunks

# ====================== Embedding Model Initialization ======================
def initialize_embedding_model():
    """
    Initialize the embedding model that converts text to vectors.
    
    What are embeddings?
    - Embeddings are numerical representations of text (vectors)
    - Similar text = similar vectors
    - This allows us to find relevant documents by comparing vectors
    
    Two options:
    1. FastEmbed: Faster but needs to be installed separately
    2. HuggingFace: Slower but more reliable (used as fallback)
    
    The model is stored in the global variable 'embedding_model'.
    """
    global embedding_model
    
    # Use device from configuration
    device = config.EMBEDDING_DEVICE
    logger.info(f"🔧 Initializing embeddings on device: {device}")
    
    # Try to use FastEmbed first (if configured and available)
    if config.USE_FASTEMBED and FASTEMBED_AVAILABLE:
        logger.info(f"🔧 Loading FastEmbed model: {config.EMBEDDING_MODEL}")
        try:
            embedding_model = FastEmbedWrapper(model_name=config.EMBEDDING_MODEL)
            logger.info("✅ FastEmbed model loaded successfully")
            return
        except Exception as e:
            logger.warning(f"FastEmbed failed, falling back to HuggingFace: {e}")
    
    # Fallback to HuggingFace embeddings (default option)
    logger.info(f"🔧 Loading HuggingFace model: {config.EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,              # Which model to use
        model_kwargs={"device": device},                 # Use GPU if available
        encode_kwargs={"normalize_embeddings": True}     # Normalize vectors for better comparison
    )
    logger.info(f"✅ HuggingFace embeddings loaded on {device}")

class FastEmbedWrapper:
    """
    Wrapper class to make FastEmbed work with LangChain.
    
    LangChain expects two methods: embed_documents() and embed_query()
    This wrapper provides those methods for FastEmbed.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize FastEmbed model."""
        self.model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple documents to embeddings (vectors).
        Used when building the FAISS index.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings (each embedding is a list of numbers)
        """
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Convert a single query to an embedding (vector).
        Used when searching for relevant documents.
        
        Args:
            text: Query string to embed
            
        Returns:
            Single embedding (list of numbers)
        """
        embedding = list(self.model.embed([text]))[0]
        return embedding.tolist()

# ====================== FAISS Index Management with Caching ======================
def save_vectorstore(vectorstore, path: str = None):
    """
    Save the FAISS vector database to disk for faster startup next time.
    
    Why save the index?
    - Building embeddings is slow (especially with many documents)
    - Saving lets us reload the index instantly next time
    - No need to re-process documents if they haven't changed
    
    Args:
        vectorstore: FAISS vector database to save
        path: Where to save (defaults to config.FAISS_INDEX_PATH)
        
    Returns:
        True if saved successfully, False otherwise
    """
    if not vectorstore:
        return False
    
    save_path = path or config.FAISS_INDEX_PATH
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the FAISS index to disk
        vectorstore.save_local(save_path)
        logger.info(f"✅ Vector store saved to {save_path}")
        
        # Save metadata about which documents were indexed
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(document_metadata, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")
        return False

def load_vectorstore(path: str = None) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.
    
    This is much faster than rebuilding the index from scratch.
    
    Args:
        path: Where to load from (defaults to config.FAISS_INDEX_PATH)
        
    Returns:
        FAISS vectorstore if found, None otherwise
    """
    load_path = path or config.FAISS_INDEX_PATH
    
    # Check if saved index exists
    if not os.path.exists(load_path):
        logger.info(f"No existing index found at {load_path}")
        return None
    
    try:
        logger.info(f"Loading existing FAISS index from {load_path}...")
        
        # Load the FAISS index
        vectorstore = FAISS.load_local(
            load_path,
            embedding_model,
            allow_dangerous_deserialization=True  # Required for loading pickled data
        )
        
        # Load metadata if available
        metadata_path = os.path.join(load_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                global document_metadata
                document_metadata = json.load(f)
        
        logger.info("✅ Loaded existing FAISS index successfully")
        return vectorstore
        
    except Exception as e:
        logger.warning(f"Failed to load index: {e}")
        return None

def should_rebuild_index(folder_path: str) -> bool:
    """
    Check if the index needs to be rebuilt.
    
    Rebuilding is needed when:
    - PDFs have been added or removed
    - First time running (no metadata exists)
    
    Args:
        folder_path: Path to PDF folder
        
    Returns:
        True if rebuild needed, False if existing index is still valid
    """
    if not document_metadata:
        return True
    
    # Get current PDF files
    current_files = set(glob.glob(os.path.join(folder_path, "*.pdf")))
    
    # Get previously indexed files
    indexed_files = set(document_metadata.get('indexed_files', []))
    
    # If files have changed, rebuild is needed
    if current_files != indexed_files:
        logger.info("Document changes detected. Index rebuild required.")
        return True
    
    return False

# ====================== Cross-Encoder Re-ranking ======================
def initialize_reranker():
    """
    Initialize the cross-encoder model for re-ranking search results.
    
    What is re-ranking?
    - Initial search (FAISS) finds similar documents quickly but not perfectly
    - Re-ranker uses a more sophisticated model to rescore the results
    - This improves accuracy by promoting truly relevant documents
    
    The re-ranker is slower but more accurate than the initial embedding search.
    """
    global reranker
    
    # Check if re-ranking is enabled
    if not config.USE_RERANKER:
        logger.warning("⚠️  Cross-encoder re-ranking is DISABLED (USE_RERANKER=false)")
        logger.info("   Set USE_RERANKER=true in .env when GPU issues are resolved")
        reranker = None
        return
    
    try:
        device = config.RERANK_DEVICE
        logger.info(f"🔧 Loading cross-encoder on device: {device}...")
        # Load a pre-trained cross-encoder model optimized for ranking
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        logger.info(f"✅ Cross-encoder loaded successfully on {device}")
    except Exception as e:
        logger.error(f"❌ Failed to load cross-encoder: {e}")
        logger.warning("⚠️  Re-ranking will be DISABLED for this session")
        reranker = None

def rerank_results(query: str, docs: List[Document]) -> tuple:
    """
    Re-rank documents to improve retrieval quality.
    
    How it works:
    1. Takes query + each document
    2. Scores how relevant each document is to the query
    3. Sorts documents by relevance score (highest first)
    
    Args:
        query: User's question
        docs: Retrieved documents from initial search
        
    Returns:
        Tuple of (sorted documents, relevance scores)
    """
    # If no documents or no reranker, return as-is
    if not docs:
        return docs, []
    
    if not reranker:
        logger.debug("Skipping re-ranking (reranker not available)")
        return docs, [0.0] * len(docs)
    
    try:
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, d.page_content] for d in docs]
        
        # Get relevance scores from the cross-encoder
        scores = reranker.predict(pairs)
        
        logger.debug(f"Reranker scores: {[f'{s:.3f}' for s in scores[:5]]}")
        
        # Sort documents by score (highest to lowest)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Separate documents and scores
        return [doc for doc, _ in ranked], [score for _, score in ranked]
        
    except Exception as e:
        logger.error(f"Re-ranking error: {e}")
        return docs, [0.0] * len(docs)

# ====================== RAG System Initialization ======================
def initialize_rag_system(force_rebuild: bool = False):
    """
    Initialize all RAG components. This is the main setup function.
    
    Steps performed:
    1. Initialize embedding model (converts text to vectors)
    2. Initialize re-ranker (improves search results)
    3. Load existing index from disk (if available) OR
    4. Build new index from PDF documents
    5. Save index to disk for future use
    
    Args:
        force_rebuild: If True, rebuild index even if cached version exists
        
    Returns:
        True if RAG system ready, False if initialization failed
    """
    global vectorstore, retriever, document_metadata, _is_rebuilding
    
    # Check if rebuild already in progress
    if force_rebuild and _is_rebuilding:
        logger.warning("⚠️  Index rebuild already in progress, skipping duplicate request")
        return False
    
    # Acquire lock for rebuild operations
    if force_rebuild:
        with _rebuild_lock:
            _is_rebuilding = True
            logger.info("🔒 Index rebuild lock acquired")
            try:
                result = _do_initialize_rag_system(force_rebuild)
                return result
            finally:
                _is_rebuilding = False
                logger.info("🔓 Index rebuild lock released")
    else:
        return _do_initialize_rag_system(force_rebuild)

def _do_initialize_rag_system(force_rebuild: bool = False):
    """
    Internal function that performs the actual RAG initialization.
    Separated from initialize_rag_system for lock management.
    """
    global vectorstore, retriever, document_metadata
    
    logger.info("🚀 Initializing RAG System...")
    
    # Step 1: Initialize embedding model (required for vector operations)
    initialize_embedding_model()
    
    # Step 2: Initialize re-ranker (improves search accuracy)
    initialize_reranker()
    
    # Step 3: Try to load existing index (unless force_rebuild is True)
    if not force_rebuild:
        vectorstore = load_vectorstore()
        if vectorstore:
            # Configure retriever to search the vectorstore
            retriever = vectorstore.as_retriever(
                search_type="similarity",      # Find similar documents
                search_kwargs={"k": config.TOP_K}  # Return top K results
            )
            logger.info("✅ RAG system initialized from cache")
            return True
    
    # Step 4: No cached index found, need to build from scratch
    # Load PDF documents using Docling
    documents = load_pdfs_with_docling(config.FOLDER_PATH)
    
    if not documents:
        logger.warning("🔧 RAG system not initialized - no documents available")
        return False
    
    # Split documents into chunks
    chunks = chunk_documents_hybrid(documents)
    
    if not chunks:
        logger.error("❌ No chunks created from documents")
        return False
    
    # Step 5: Build FAISS vector index
    try:
        logger.info("🔧 Building FAISS vector index...")
        
        # Create FAISS index from document chunks
        # This generates embeddings for all chunks and stores them
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        
        # Configure retriever to search the vectorstore
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K}
        )
        
        # Save metadata about indexed documents
        document_metadata = {
            'indexed_files': [doc['metadata']['source'] for doc in documents],
            'num_chunks': len(chunks),
            'indexed_at': datetime.now().isoformat()
        }
        
        # Save index to disk for next time
        save_vectorstore(vectorstore)
        
        logger.info("✅ RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing RAG system: {e}", exc_info=True)
        logger.info("🔧 Falling back to LLM-only mode")
        return False

# ====================== Ollama LLM Interface with Retry Logic ======================
def create_session_with_retries():
    """
    Create HTTP session with automatic retry logic.
    
    Why retries?
    - Network requests can fail temporarily
    - Ollama server might be busy
    - Automatic retries make the system more robust
    
    Returns:
        requests.Session configured with retry strategy
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,                                      # Try up to 3 times
        backoff_factor=1,                             # Wait 1s, 2s, 4s between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP errors
        allowed_methods=["POST"]                      # Only retry POST requests
    )
    
    # Apply retry strategy to session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Create global session for reuse
session = create_session_with_retries()

def query_local_ollama(
    prompt: str,
    model_name: str = None,
    url: str = None,
    max_tokens: int = None,
    temperature: float = None
) -> Optional[str]:
    """
    Send a prompt to Ollama and get the AI-generated response.
    
    Ollama is a local LLM server that runs models like Llama, Qwen, etc.
    
    Args:
        prompt: The text prompt to send to the AI
        model_name: Which model to use (e.g., "qwen3:7b")
        url: Ollama API endpoint URL
        max_tokens: Maximum length of response
        temperature: Randomness (0=deterministic, 1=creative)
        
    Returns:
        Generated text response, or None if request failed
    """
    # Use defaults from config if not specified
    model_name = model_name or config.OLLAMA_MODEL
    url = url or config.OLLAMA_API_URL
    max_tokens = max_tokens or config.MAX_RESPONSE_TOKENS
    temperature = temperature if temperature is not None else config.TEMPERATURE
    
    # Prepare request payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": []  # List of strings that stop generation
    }
    
    try:
        # Send POST request to Ollama
        response = session.post(url, json=payload, timeout=120)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Extract generated text from response
        # Different Ollama versions use different response formats
        if "completion" in data:
            return data["completion"].strip()
        elif "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        else:
            logger.warning(f"Unexpected response format: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ollama: {e}")
        return None

# ====================== RAG Query Functions ======================
def rag_query(query: str) -> tuple:
    """
    HYBRID MODE: Answer question using both documents (RAG) and AI knowledge.
    
    This is the most flexible mode:
    - Searches documents for relevant information
    - Uses AI to synthesize an answer combining document info + general knowledge
    - Falls back gracefully if no relevant documents found
    
    Process:
    1. Search vector database for similar chunks
    2. Re-rank results to get most relevant
    3. Build prompt with document context
    4. Generate answer using Ollama
    5. Clean up the response
    
    Args:
        query: User's question
        
    Returns:
        Tuple of (answer: str, context_used: bool, sources: List[str])
    """
    context_text = ""
    context_used = False
    sources = []
    
    # Try to retrieve relevant documents if RAG is available
    if vectorstore and retriever:
        try:
            # Search for similar documents with similarity scores
            results_with_scores = vectorstore.similarity_search_with_score(query, k=config.TOP_K)
            logger.info(f"🔍 Top similarity scores: {[f'{score:.3f}' for _, score in results_with_scores[:5]]}")
            
            if results_with_scores:
                # Extract just the documents (without scores)
                retrieved_docs = [doc for doc, score in results_with_scores]
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                # Re-rank documents for better relevance
                reranked_docs, rerank_scores = rerank_results(query, retrieved_docs)
                
                # Take top N documents after re-ranking
                top_docs = reranked_docs[:config.RERANK_TOP_K]
                
                # Build context from top documents
                context_blocks = []
                for doc in top_docs:
                    filename = doc.metadata.get("filename", "unknown.pdf")
                    chunk_id = doc.metadata.get("chunk_id", "N/A")
                    # Format: content + source citation
                    context_blocks.append(
                        f"{doc.page_content}\n[Source: {filename}, Chunk {chunk_id}]"
                    )
                    # Track sources for metadata
                    sources.append(f"{filename} (Chunk {chunk_id})")
                
                context_text = "\n\n".join(context_blocks)
                context_used = True  # Mark that we used document context
                
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)
    
    # Build prompt based on whether we have context
    if context_text.strip():
        # Prompt with document context
        prompt = f"""Answer this question in 3-4 sentences. Use the documents if helpful.

Documents:
{context_text}

Question: {query}

Answer:"""
    else:
        # Prompt without context (pure AI knowledge)
        prompt = f"""Answer this question in 3-4 sentences.

Question: {query}

Answer:"""
    
    # Generate answer using Ollama (limited tokens for concise answers)
    answer = query_local_ollama(prompt, max_tokens=200, temperature=0.1)
    
    # Clean up the answer (remove artifacts and extra paragraphs)
    if answer:
        # Remove common stop patterns that leak through
        for stop_pattern in ["<|endoftext|>", "Write a", "[Source:", "Note:"]:
            if stop_pattern in answer:
                answer = answer.split(stop_pattern)[0].strip()
        
        # Keep only first paragraph (sometimes model generates too much)
        paragraphs = answer.split('\n\n')
        if len(paragraphs) > 1:
            answer = paragraphs[0].strip()
    
    # Log what happened for debugging
    logger.info("\n======== Thinking ========")
    if context_text.strip():
        logger.info(f"Context used:\n{context_text[:500]}...")
    else:
        logger.info("No RAG context. Using LLM general knowledge.")
    
    logger.info("\n======== Answer ========")
    return (answer or "⚠️ LLM did not return an answer."), context_used, sources

def llm_only_query(query: str) -> tuple:
    """
    LLM-ONLY MODE: Answer question using only the AI's general knowledge.
    
    No document retrieval - just pure AI reasoning.
    Useful for:
    - General knowledge questions
    - Testing AI capabilities
    - When documents don't contain relevant info
    
    Args:
        query: User's question
        
    Returns:
        AI-generated answer (string)
    """
    # Simple prompt without any document context
    prompt = f"""Answer this question in 3-4 sentences.

Question: {query}

Answer:"""
    
    # Generate answer using Ollama
    answer = query_local_ollama(prompt, max_tokens=200, temperature=0.1)
    
    # Clean up the answer
    if answer:
        # Remove artifacts
        for stop_pattern in ["<|endoftext|>", "Write a", "Question:"]:
            if stop_pattern in answer:
                answer = answer.split(stop_pattern)[0].strip()
        
        # Keep only first paragraph
        paragraphs = answer.split('\n\n')
        if len(paragraphs) > 1:
            answer = paragraphs[0].strip()
    
    # Log for debugging
    logger.info("\n======== Thinking ========")
    logger.info("Using LLM general knowledge only (no RAG)")
    logger.info("\n======== Answer ========")
    
    return (answer or "⚠️ LLM did not return an answer."), False, []

def rag_only_query(query: str, relevance_threshold: float = 1.5) -> tuple:
    """
    RAG-ONLY MODE: Answer ONLY using document information (strict mode).
    
    This mode:
    - ONLY uses information from documents
    - Refuses to answer if no relevant documents found
    - More conservative but more trustworthy
    - Good for fact-checking against specific documents
    
    How it works:
    1. Search documents with similarity scoring
    2. Filter out documents below relevance threshold
    3. If no relevant docs, return "not found" message
    4. If relevant docs found, generate answer using ONLY that context
    
    Args:
        query: User's question
        relevance_threshold: Maximum distance score (lower = more similar)
                            Default 1.5 works well for most cases
        
    Returns:
        Tuple of (answer: str, context_used: bool, sources: List[str])
    """
    if not vectorstore:
        logger.warning("No RAG system available")
        return "⚠️ No documents available. Please add PDFs to enable RAG.", False, []
    
    try:
        # Search for similar documents with scores
        # Score = distance (lower is better, 0 = perfect match)
        results = vectorstore.similarity_search_with_score(query, k=10)
        logger.info(f"🔍 Similarity scores: {[f'{score:.3f}' for _, score in results[:5]]}")
        
        # Filter documents by relevance threshold
        relevant_docs = [(doc, score) for doc, score in results if score < relevance_threshold]
        
        # If no relevant documents found, return error message
        if not relevant_docs:
            logger.warning(f"No relevant documents found (best: {results[0][1]:.3f})")
            return f"⚠️ No relevant information found in documents for: '{query}'", False, []
        
        # Get top documents
        top_docs = [doc for doc, score in relevant_docs[:config.TOP_K]]
        
        # Re-rank for better accuracy
        reranked_docs, _ = rerank_results(query, top_docs)
        final_docs = reranked_docs[:config.RERANK_TOP_K]
        
        # Build context from documents
        context_blocks = []
        sources = []
        for doc in final_docs:
            filename = doc.metadata.get("filename", "unknown.pdf")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            context_blocks.append(
                f"{doc.page_content}\n[Source: {filename}, Chunk {chunk_id}]"
            )
            # Track sources for metadata
            sources.append(f"{filename} (Chunk {chunk_id})")
        
        context_text = "\n\n".join(context_blocks)
        
        # IMPORTANT: Very directive prompt to prevent hallucination
        prompt = f"""Answer this question in 2-3 sentences using ONLY the information below.

Documents:
{context_text}

Question: {query}

Answer:"""
        
        # CRITICAL: Low token limit to prevent model from adding extra info
        answer = query_local_ollama(prompt, max_tokens=150, temperature=0.1)
        
        # Log what documents were used
        logger.info("\n======== Thinking ========")
        logger.info(f"Found {len(final_docs)} relevant documents:")
        logger.info(context_text[:500])
        logger.info("\n======== Answer ========")
        
        # Clean up any unwanted text that leaked through
        if answer:
            # Remove everything after common stop patterns
            for stop_pattern in ["<|endoftext|>", "Write a", "[Source:", "(Source:", "Note:", "[Note:"]:
                if stop_pattern in answer:
                    answer = answer.split(stop_pattern)[0].strip()
            
            # Ensure only the first paragraph is returned
            paragraphs = answer.split('\n\n')
            if len(paragraphs) > 1:
                answer = paragraphs[0].strip()
        
        return (answer or "⚠️ LLM did not return an answer."), True, sources
        
    except Exception as e:
        logger.error(f"RAG-only query error: {e}", exc_info=True)
        return f"⚠️ Error searching documents: {str(e)}", False, []

# ====================== FastAPI Application Setup ======================
app = FastAPI(
    title="Enhanced RAG Pipeline API",
    description="Production-ready RAG system with Docling integration",
    version="2.0"
)

# ====================== CORS Middleware for Open WebUI ======================
from fastapi.middleware.cors import CORSMiddleware

# Allow Open WebUI to connect from any origin
# This is essential for browser-based clients like Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Open WebUI URL: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== API Authentication ======================
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify API key from Authorization header.
    
    If no API key is configured in .env, allows all requests (development mode).
    If API key is configured, validates it against the provided credentials.
    
    Args:
        credentials: HTTP Bearer token from Authorization header
        
    Returns:
        True if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    # If no API key configured, allow all requests (dev mode)
    if not config.API_KEY:
        return True
    
    # Check if credentials provided
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Set Authorization: Bearer <key> header"
        )
    
    # Verify key matches
    if credentials.credentials != config.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when FastAPI starts"""
    logger.info("=" * 80)
    logger.info("🚀 Starting RAG System Initialization...")
    logger.info("=" * 80)
    
    try:
        initialize_rag_system()
        logger.info("✅ RAG system initialization complete!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}", exc_info=True)
        logger.warning("⚠️ Server will start but RAG functionality may be unavailable")


# ====================== OpenAI-Compatible Chat Endpoint ======================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    authorized: bool = Depends(verify_api_key)
):
    """
    OpenAI-compatible chat completion endpoint for Open WebUI.
    This intercepts queries and uses your enhanced RAG pipeline.
    """
    import time
    start_time = time.time()
    
    try:
        # Extract the user's latest message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        user_query = user_messages[-1].content
        logger.info(f"📨 Received query from Open WebUI: {user_query[:100]}...")
        
        # Determine query mode based on message content
        # You can add logic here to detect if RAG is needed
        if should_use_rag(user_query):
            logger.info("🔍 Using RAG mode")
            answer, context_used, sources = rag_query(user_query)
            mode = "rag"
        else:
            logger.info("💬 Using LLM-only mode")
            answer, context_used, sources = llm_only_query(user_query)
            mode = "llm-only"
        
        # Add visibility note if no context was used
        if not context_used and mode == "rag":
            answer = f"{answer}\n\n_(No company documents were used for this answer)_"
        
        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_query.split()),
                "completion_tokens": len(answer.split()),
                "total_tokens": len(user_query.split()) + len(answer.split())
            }
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Query processed in {processing_time:.2f}s using {mode} mode")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def should_use_rag(query: str) -> bool:
    """
    Determine if query should use RAG or just LLM.
    
    Returns True if:
    - Query mentions documents/files
    - Query asks about specific topics in your PDFs
    - Default to True for now (always use RAG)
    """
    # For now, always use RAG if documents are loaded
    if vectorstore is not None:
        return True
    return False

# ====================== FastAPI Integration ======================
# FastAPI allows this script to be used as a web service (REST API)
# Other applications can send HTTP requests to get answers

class QueryRequest(BaseModel):
    """
    Schema for API query requests.
    Validates incoming requests to ensure they have required fields.
    """
    question: str = Field(..., min_length=1, description="User question")
    mode: str = Field(default="hybrid", pattern="^(hybrid|rag_only|llm_only)$")

class QueryResponse(BaseModel):
    """
    Schema for API query responses.
    Defines what information is returned.
    """
    answer: str                               # The generated answer
    mode_used: str                            # Which mode was used
    context_available: bool                   # Whether RAG context was available
    processing_time: Optional[float] = None   # How long it took (seconds)

@app.post("/query", response_model=QueryResponse)
async def api_query(
    request: QueryRequest,
    authorized: bool = Depends(verify_api_key)
):
    """
    API endpoint for querying the RAG system.
    
    Usage:
        POST /query
        Body: {"question": "What is X?", "mode": "hybrid"}
    
    Returns:
        JSON response with answer and metadata
    """
    import time
    start_time = time.time()
    
    try:
        question = request.question.strip()
        mode = request.mode.lower()
        
        # Determine which query function to use based on mode
        if mode == "rag_only" and vectorstore:
            answer, context_used, sources = rag_only_query(question)
            mode_used = "rag_only"
            context_available = True
        elif mode == "llm_only":
            answer, context_used, sources = llm_only_query(question)
            mode_used = "llm_only"
            context_available = False
        else:  # hybrid mode (default)
            if vectorstore:
                answer, context_used, sources = rag_query(question)
                mode_used = "hybrid"
                context_available = True
            else:
                # No RAG available, fall back to LLM only
                answer, context_used, sources = llm_only_query(question)
                mode_used = "llm_only"
                context_available = False
        
        # Calculate how long it took
        processing_time = time.time() - start_time
        
        # Return formatted response
        return QueryResponse(
            answer=answer,
            mode_used=mode_used,
            context_available=context_available,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"API query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the system is running.
    
    Usage:
        GET /health
    
    Returns:
        System status and configuration info
    """
    return {
        "status": "healthy",
        "rag_available": vectorstore is not None,
        "model": config.OLLAMA_MODEL,
        "documents_indexed": document_metadata.get('num_chunks', 0),
        "embedding_model": config.EMBEDDING_MODEL
    }

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint.
    Open WebUI uses this to discover available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": config.OLLAMA_MODEL,
                "object": "model",
                "created": 1699564800,
                "owned_by": "ollama",
                "permission": [],
                "root": config.OLLAMA_MODEL,
                "parent": None
            }
        ]
    }


@app.post("/rebuild-index")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    authorized: bool = Depends(verify_api_key)
):
    """
    Trigger a rebuild of the FAISS index.
    
    Useful when:
    - New PDFs have been added
    - Documents have been modified
    - Index got corrupted
    
    The rebuild happens in the background so the API responds immediately.
    
    Usage:
        POST /rebuild-index
    """
    # Check if already rebuilding
    if _is_rebuilding:
        return {
            "message": "Index rebuild already in progress",
            "status": "busy"
        }
    
    background_tasks.add_task(initialize_rag_system, force_rebuild=True)
    return {
        "message": "Index rebuild started in background",
        "status": "started"
    }

@app.get("/rebuild-status")
async def rebuild_status():
    """
    Check the status of index rebuild operations.
    No authentication required for status checks.
    
    Usage:
        GET /rebuild-status
    
    Returns:
        Current rebuild status and document count
    """
    return {
        "is_rebuilding": _is_rebuilding,
        "documents_indexed": document_metadata.get('num_chunks', 0),
        "last_indexed": document_metadata.get('indexed_at', 'never')
    }

# ====================== Main Interactive Loop ======================
def main():
    """
    Main function for interactive command-line usage.
    
    This creates an interactive loop where you can:
    - Type questions and get answers
    - Switch between different modes (hybrid, rag_only, llm_only)
    - Rebuild the index on demand
    - Exit when done
    
    The loop will keep running until you type 'exit' or 'quit'.
    """
    logger.info("🚀 Starting Enhanced RAG Pipeline System...")
    
    # Initialize the RAG system
    rag_available = initialize_rag_system()
    
    # Show startup message based on whether RAG is available
    if rag_available:
        logger.info("\n✅ System ready with RAG capabilities!")
        logger.info("📋 Query modes:")
        logger.info("   - Default: Hybrid RAG + LLM")
        logger.info("   - 'rag_only:' prefix for strict RAG")
        logger.info("   - 'llm_only:' prefix for pure LLM")
        default_function = rag_query
    else:
        logger.info("\n✅ System ready in LLM-only mode!")
        default_function = llm_only_query
    
    # Interactive loop - keeps asking for questions
    while True:
        try:
            # Get user input
            question = input("\n🔍 Enter your question (or 'exit'/'rebuild' to quit/rebuild): ").strip()
            
            # Check for exit command
            if question.lower() in ["exit", "quit"]:
                logger.info("👋 Exiting. Goodbye!")
                break
            
            # Check for rebuild command
            if question.lower() == "rebuild":
                logger.info("Rebuilding index...")
                initialize_rag_system(force_rebuild=True)
                continue
            
            # Validate input
            if not question:
                logger.warning("Please enter a valid question")
                continue
            
            # Parse mode prefix (if user specified a mode)
            query_function = default_function
            
            if question.startswith("rag_only:"):
                # User wants strict RAG mode
                question = question[9:].strip()  # Remove prefix
                query_function = rag_only_query if rag_available else llm_only_query
                
            elif question.startswith("llm_only:"):
                # User wants LLM-only mode
                question = question[9:].strip()  # Remove prefix
                query_function = llm_only_query
                
            elif question.startswith("hybrid:"):
                # User explicitly wants hybrid mode
                question = question[7:].strip()  # Remove prefix
                query_function = rag_query if rag_available else llm_only_query
            
            # Display the question
            logger.info(f"\nQuestion: {question}")
            logger.info("=" * 50)
            
            # Generate and display answer
            answer, context_used, sources = query_function(question)
            print("\n" + "=" * 50)
            print(answer)
            print("=" * 50)
            
            # Show metadata about context usage
            if context_used and sources:
                # Extract unique document names (remove chunk numbers)
                unique_docs = []
                for source in sources:
                    # Extract just the filename (before " (Chunk")
                    doc_name = source.split(" (Chunk")[0]
                    if doc_name not in unique_docs:
                        unique_docs.append(doc_name)
                
                print("\n✅ Answer based on your documents")
                print("\n📚 Sources:")
                for i, doc in enumerate(unique_docs, 1):
                    print(f"   {i}. {doc}")
            elif context_used:
                print("\n✅ Answer based on your documents")
            else:
                print("\n⚠️  No documents used - answer from AI general knowledge")
            
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            logger.info("\n👋 Interrupted. Goodbye!")
            break
            
        except Exception as e:
            # Handle unexpected errors gracefully
            logger.error(f"Error processing question: {e}", exc_info=True)
            continue

# ====================== Entry Point ======================
if __name__ == "__main__":
    import sys
    
    # Check if running as API server or interactive CLI
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Run as FastAPI server (for Docker)
        import uvicorn
        
        # Initialize RAG system on startup
        logger.info("Initializing RAG system for API server mode...")
        initialize_rag_system()
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        # Run interactive CLI (your current behavior)
        main()

