import chromadb
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, persist_directory: str = "db"):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[str], embeddings: List[List[float]]):
        try:
            if not documents or not embeddings:
                logger.warning("Empty documents or embeddings provided")
                return
                
            if len(documents) != len(embeddings):
                raise ValueError(f"Number of documents ({len(documents)}) does not match number of embeddings ({len(embeddings)})")
                
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to the collection")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        try:
            if not query_embedding:
                raise ValueError("Query embedding is empty")
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results or 'documents' not in results or not results['documents']:
                logger.warning("No results found for query")
                return []
                
            logger.info(f"Found {len(results['documents'][0])} relevant documents")
            return results['documents'][0]
            
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            raise

    def reset(self):
        try:
            self.client.delete_collection(name="document_collection")
            self.collection = self.client.create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Error resetting ChromaDB: {e}")
            raise