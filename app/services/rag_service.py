"""
RAG (Retrieval-Augmented Generation) Service

This service provides RAG functionality using vector database retrieval
and LLM-based generation. Supports both local and cloud-based embeddings/LLMs.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. RAG will use fallback mode.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. RAG will use fallback mode.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. RAG will use fallback mode.")


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations."""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the RAG service.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model name
            openai_api_key: OpenAI API key (optional, for LLM generation)
            openai_model: OpenAI model to use for generation
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or Path(__file__).parent.parent / "data" / "chroma"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.vector_db = None
        self.collection = None
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.openai_client = None
        
        self.is_initialized = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector database and embedding model."""
        try:
            # Initialize ChromaDB
            if CHROMADB_AVAILABLE:
                self.vector_db = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.vector_db.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
            else:
                logger.warning("ChromaDB not available - using fallback mode")
            
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            else:
                logger.warning("sentence-transformers not available - using fallback mode")
            
            # Initialize OpenAI client if available
            if OPENAI_AVAILABLE and self.openai_api_key:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI API configured")
            else:
                self.openai_client = None
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            self.is_initialized = False
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if unavailable
        """
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                return None
        return None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents with 'text' and optional 'metadata'
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized or not self.collection:
            logger.warning("RAG service not initialized - cannot add documents")
            return False
        
        try:
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                text = doc.get("text", "")
                if not text:
                    continue
                
                texts.append(text)
                metadatas.append(doc.get("metadata", {}))
                ids.append(doc.get("id", f"doc_{i}_{hash(text)}"))
            
            if not texts:
                logger.warning("No valid documents to add")
                return False
            
            # Generate embeddings
            embeddings = []
            for text in texts:
                embedding = self._get_embedding(text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
                    return False
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} documents to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: User query
            n_results: Number of relevant documents to retrieve
            max_tokens: Maximum tokens in generated response
            temperature: Sampling temperature for LLM
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        if not self.is_initialized:
            return self._fallback_response(query_text)
        
        try:
            # Retrieve relevant documents
            relevant_docs = self._retrieve_documents(query_text, n_results)
            
            # Generate response using LLM
            response = self._generate_response(query_text, relevant_docs, max_tokens, temperature)
            
            # Extract sources
            sources = [doc.get("metadata", {}).get("source", "unknown") for doc in relevant_docs]
            sources = [s for s in sources if s != "unknown"] or None
            
            return {
                "query": query_text,
                "response": response,
                "sources": sources if sources else None,
                "confidence": 0.85 if relevant_docs else 0.5,
                "retrieved_docs": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return self._fallback_response(query_text)
    
    def _retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector database.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.collection or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            documents = []
            if results and results.get("documents"):
                for i, doc_text in enumerate(results["documents"][0]):
                    doc = {
                        "text": doc_text,
                        "metadata": results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {},
                        "distance": results.get("distances", [[]])[0][i] if results.get("distances") else None
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _generate_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved relevant documents
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        # Build context from retrieved documents
        context = "\n\n".join([doc.get("text", "") for doc in context_docs])
        
        # Use OpenAI if available
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on the provided context. "
                                     "If the context doesn't contain relevant information, say so."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}, using fallback")
        
        # Fallback: simple template-based response
        if context:
            return (
                f"Based on the available information:\n\n{context[:max_tokens//2]}\n\n"
                f"This is a summary related to your question: '{query}'. "
                "For a more detailed response, please configure OpenAI API."
            )
        else:
            return (
                f"I don't have specific information about '{query}' in my knowledge base. "
                "Please configure document ingestion and OpenAI API for full RAG functionality."
            )
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate a fallback response when RAG is not fully initialized."""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if "phishing" in query_lower or "security" in query_lower:
            response = (
                "Phishing is a type of cyber attack where attackers attempt to trick users "
                "into revealing sensitive information by impersonating legitimate entities. "
                "Common indicators include suspicious URLs, unexpected requests for credentials, "
                "and urgent language designed to create panic."
            )
            confidence = 0.75
        elif "api" in query_lower or "endpoint" in query_lower:
            response = (
                "This API provides endpoints for phishing detection and RAG-based question answering. "
                "The phishing endpoint accepts URLs and returns classification results. "
                "The RAG endpoint processes natural language queries and returns contextual responses."
            )
            confidence = 0.80
        else:
            response = (
                f"This is a fallback response to your query: '{query}'. "
                "The RAG system requires ChromaDB and sentence-transformers to be installed. "
                "Install them with: pip install chromadb sentence-transformers"
            )
            confidence = 0.50
        
        return {
            "query": query,
            "response": response,
            "sources": None,
            "confidence": confidence,
            "retrieved_docs": 0
        }
    
    def initialize_with_sample_data(self) -> bool:
        """Initialize the RAG system with sample cybersecurity documents."""
        sample_docs = [
            {
                "id": "doc_phishing_1",
                "text": (
                    "Phishing is a cyber attack method where attackers send fraudulent communications "
                    "that appear to come from a reputable source, usually through email. The goal is to "
                    "steal sensitive data like login credentials, credit card numbers, or install malware "
                    "on the victim's machine. Common phishing indicators include: suspicious sender addresses, "
                    "urgent language, requests for sensitive information, and suspicious links or attachments."
                ),
                "metadata": {"source": "cybersecurity_guide.pdf", "topic": "phishing", "type": "educational"}
            },
            {
                "id": "doc_phishing_2",
                "text": (
                    "Phishing detection techniques include: checking URL authenticity, verifying sender identity, "
                    "examining email headers, looking for spelling and grammar errors, and being cautious of "
                    "unsolicited requests. Machine learning models can analyze URL features, email content, "
                    "and behavioral patterns to automatically detect phishing attempts."
                ),
                "metadata": {"source": "phishing_detection_handbook.pdf", "topic": "phishing", "type": "technical"}
            },
            {
                "id": "doc_security_1",
                "text": (
                    "API security best practices include: implementing authentication and authorization, "
                    "using HTTPS for all communications, validating and sanitizing all inputs, implementing "
                    "rate limiting to prevent abuse, logging and monitoring API access, and keeping "
                    "dependencies up to date."
                ),
                "metadata": {"source": "api_security_guide.pdf", "topic": "api_security", "type": "best_practices"}
            },
            {
                "id": "doc_rag_1",
                "text": (
                    "RAG (Retrieval-Augmented Generation) combines information retrieval with language models. "
                    "It retrieves relevant documents from a knowledge base and uses them as context for "
                    "generating accurate, contextual responses. RAG systems typically use vector databases "
                    "for semantic search and large language models for text generation."
                ),
                "metadata": {"source": "rag_technical_guide.pdf", "topic": "rag", "type": "technical"}
            }
        ]
        
        return self.add_documents(sample_docs)


# Global service instance
_rag_service: Optional['RAGService'] = None


def get_rag_service() -> RAGService:
    """
    Get or create the global RAG service instance.
    
    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        # Initialize with sample data if collection is empty
        try:
            if _rag_service.is_initialized and _rag_service.collection:
                count = _rag_service.collection.count()
                if count == 0:
                    logger.info("Initializing RAG service with sample data...")
                    _rag_service.initialize_with_sample_data()
        except Exception as e:
            logger.warning(f"Could not check/initialize sample data: {e}")
    return _rag_service


def initialize_rag_service(
    collection_name: Optional[str] = None,
    persist_directory: Optional[Path] = None,
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> bool:
    """
    Initialize the global RAG service.
    
    Args:
        collection_name: Optional collection name
        persist_directory: Optional persist directory
        embedding_model: Optional embedding model name
        openai_api_key: Optional OpenAI API key
        
    Returns:
        True if initialization successful, False otherwise
    """
    global _rag_service
    kwargs = {}
    if collection_name:
        kwargs["collection_name"] = collection_name
    if persist_directory:
        kwargs["persist_directory"] = persist_directory
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    if openai_api_key:
        kwargs["openai_api_key"] = openai_api_key
    
    _rag_service = RAGService(**kwargs)
    return _rag_service.is_initialized

