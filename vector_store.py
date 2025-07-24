import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
import os
from document_processor import DocumentChunk

class VectorStore:
    """Vector store for document embeddings using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger.info(f"Initialized vector store with {self.collection.count()} documents")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            bool: Success status
        """
        try:
            if not chunks:
                return True
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                documents.append(chunk.text)
                metadatas.append(chunk.metadata)
                ids.append(str(uuid.uuid4()))
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(chunks)} document chunks to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_metadata: Optional[Dict] = None) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of tuples (document_text, metadata, score)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'][0] else 0.0
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance
                    search_results.append((doc, metadata, score))
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def search_by_paper(self, paper_title: str, query: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        Search within a specific paper
        
        Args:
            paper_title: Title of the paper to search within
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (document_text, metadata, score)
        """
        filter_metadata = {"title": {"$eq": paper_title}}
        return self.similarity_search(query, k, filter_metadata)
    
    def get_paper_content(self, paper_title: str) -> List[Tuple[str, Dict]]:
        """
        Get all content from a specific paper
        
        Args:
            paper_title: Title of the paper
            
        Returns:
            List of tuples (document_text, metadata)
        """
        try:
            results = self.collection.get(
                where={"title": {"$eq": paper_title}}
            )
            
            content = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    content.append((doc, metadata))
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error getting paper content: {e}")
            return []
    
    def delete_paper(self, paper_title: str) -> bool:
        """
        Delete all chunks from a specific paper
        
        Args:
            paper_title: Title of the paper to delete
            
        Returns:
            bool: Success status
        """
        try:
            # Get all documents for this paper
            results = self.collection.get(
                where={"title": {"$eq": paper_title}}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                self.logger.info(f"Deleted {len(results['ids'])} chunks for paper: {paper_title}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting paper: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            
            # Get unique papers
            results = self.collection.get()
            unique_papers = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'title' in metadata:
                        unique_papers.add(metadata['title'])
            
            return {
                'total_chunks': count,
                'unique_papers': len(unique_papers),
                'papers': list(unique_papers)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {'total_chunks': 0, 'unique_papers': 0, 'papers': []}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name="research_papers")
            self.collection = self.client.create_collection(
                name="research_papers",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info("Cleared vector store collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (document_text, metadata, score)
        """
        try:
            # Semantic search
            semantic_results = self.similarity_search(query, k)
            
            # Simple keyword search (can be enhanced with full-text search)
            keyword_results = self._keyword_search(query, k)
            
            # Combine and re-rank results
            combined_results = self._combine_search_results(semantic_results, keyword_results)
            
            return combined_results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[str, Dict, float]]:
        """Simple keyword search implementation"""
        try:
            # Get all documents
            results = self.collection.get()
            
            if not results['documents']:
                return []
            
            # Score documents based on keyword matches
            scored_results = []
            query_terms = query.lower().split()
            
            for i, doc in enumerate(results['documents']):
                doc_lower = doc.lower()
                score = 0
                
                for term in query_terms:
                    score += doc_lower.count(term)
                
                if score > 0:
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    scored_results.append((doc, metadata, score))
            
            # Sort by score and return top k
            scored_results.sort(key=lambda x: x[2], reverse=True)
            return scored_results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []
    
    def _combine_search_results(self, semantic_results: List[Tuple[str, Dict, float]], 
                               keyword_results: List[Tuple[str, Dict, float]]) -> List[Tuple[str, Dict, float]]:
        """Combine semantic and keyword search results"""
        # Simple combination strategy - can be enhanced with more sophisticated ranking
        combined = {}
        
        # Add semantic results with higher weight
        for doc, metadata, score in semantic_results:
            doc_id = hash(doc)
            combined[doc_id] = (doc, metadata, score * 0.7)
        
        # Add keyword results with lower weight
        for doc, metadata, score in keyword_results:
            doc_id = hash(doc)
            if doc_id in combined:
                # Boost score if found in both
                combined[doc_id] = (combined[doc_id][0], combined[doc_id][1], 
                                  combined[doc_id][2] + score * 0.3)
            else:
                combined[doc_id] = (doc, metadata, score * 0.3)
        
        # Sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results