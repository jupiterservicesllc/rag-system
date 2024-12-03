import pdfplumber
from typing import List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFProcessor:
    def extract_text(self, pdf_path: str, chunk_size: int = 1000) -> List[str]:
        """
        Extract text from PDF and split into chunks
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Extract text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return []
                
            # Split text into chunks
            chunks = []
            current_chunk = ""
            
            # Split by sentences (simple approach)
            sentences = text.replace("\n", " ").split(".")
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
                    
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            logger.info(f"Extracted {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise