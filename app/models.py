import os
from typing import List
import logging
from pathlib import Path
import re
from collections import Counter
import math
import numpy as np
from hashlib import sha256

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> List[str]:
    """Clean and tokenize text"""
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove punctuation and numbers
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    # Remove empty strings and single characters
    words = [word for word in words if len(word) > 1]
    return words

class EmbeddingModel:
    def __init__(self):
        """Initialize the embedding model"""
        self.vector_size = 384  # Match ChromaDB's dimension
        self.word_vectors = {}
        logger.info("Embedding model initialized")

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get a deterministic vector for a word using its hash"""
        if word not in self.word_vectors:
            # Use word's hash to generate a deterministic vector
            word_hash = sha256(word.encode()).digest()
            # Convert hash bytes to float array
            vector = np.frombuffer(word_hash[:48], dtype=np.uint8).astype(np.float32)
            # Reshape to desired size and normalize
            vector = vector / np.linalg.norm(vector)
            # Pad or truncate to match vector_size
            if len(vector) < self.vector_size:
                vector = np.pad(vector, (0, self.vector_size - len(vector)))
            else:
                vector = vector[:self.vector_size]
            self.word_vectors[word] = vector
        return self.word_vectors[word]

    def generate_embeddings(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        embeddings = []
        for doc in documents:
            words = preprocess_text(doc)
            if not words:
                embeddings.append([0.0] * self.vector_size)
                continue
                
            # Average word vectors for the document
            doc_vector = np.zeros(self.vector_size)
            for word in words:
                doc_vector += self.get_word_vector(word)
            if len(words) > 0:
                doc_vector /= len(words)
                
            # Normalize the final vector
            norm = np.linalg.norm(doc_vector)
            if norm > 0:
                doc_vector /= norm
                
            embeddings.append(doc_vector.tolist())
            
        return embeddings

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        words = preprocess_text(query)
        if not words:
            return [0.0] * self.vector_size
            
        # Average word vectors for the query
        query_vector = np.zeros(self.vector_size)
        for word in words:
            query_vector += self.get_word_vector(word)
        if len(words) > 0:
            query_vector /= len(words)
            
        # Normalize the final vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector /= norm
            
        return query_vector.tolist()

class QueryModel:
    def __init__(self):
        """Initialize the query model"""
        logger.info("Query model initialized")

    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using a rule-based approach with Q&A formatting"""
        try:
            # Combine all context documents
            combined_context = " ".join(context_docs)
            
            # Extract relevant sentences that might contain the answer
            sentences = re.split(r'[.!?]+', combined_context)
            relevant_sentences = []
            
            # Simple keyword matching
            query_words = set(preprocess_text(query))
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_words = set(preprocess_text(sentence))
                # If the sentence shares keywords with the query, consider it relevant
                if query_words & sentence_words:
                    relevant_sentences.append(sentence)
            
            # Format the response in Q&A format
            response = f"Question: {query}\n\nAnswer:\n"
            
            if not relevant_sentences:
                response += "I couldn't find a specific answer to your question in the provided documents."
                return response
            
            # Format relevant sentences as a numbered list
            response += f"{relevant_sentences[0]} Here are the key points:\n"
            for i, sentence in enumerate(relevant_sentences[1:], 1):
                response += f"{i}) {sentence.strip()}\n"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Question: {query}\n\nAnswer:\nI apologize, but I encountered an error while generating the response."