import os
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.database import VectorDatabase
from app.pdf_processor import PDFProcessor
from app.models import EmbeddingModel, QueryModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize application
app = FastAPI(title="RAG System")

# Setup templates and static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Ensure temp directory exists
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize components
vector_db = VectorDatabase()
pdf_processor = PDFProcessor()
embedding_model = EmbeddingModel()
query_model = QueryModel()

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        temp_path = TEMP_DIR / file.filename
        
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Process PDF
            text_chunks = pdf_processor.extract_text(str(temp_path))
            if not text_chunks:
                raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
                
            # Generate embeddings
            embeddings = embedding_model.generate_embeddings(text_chunks)
            
            # Store in vector database
            vector_db.add_documents(text_chunks, embeddings)
            
            return JSONResponse(content={"message": "PDF processed successfully", "chunks": len(text_chunks)})
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: str):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        logger.info(f"Processing query: {query}")
        
        # Generate query embedding
        query_embedding = embedding_model.generate_query_embedding(query)
        logger.info("Generated query embedding")
        
        # Search for relevant documents
        context_docs = vector_db.search(query_embedding)
        logger.info(f"Found {len(context_docs)} relevant documents")
        
        if not context_docs:
            return JSONResponse(content={
                "response": "I couldn't find any relevant information in the uploaded documents to answer your question."
            })
        
        # Generate response using LLM
        response = query_model.generate_response(query, context_docs)
        logger.info("Generated response")
        
        return JSONResponse(content={"response": response})
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/clear")
async def clear_database():
    try:
        vector_db.reset()
        return JSONResponse(content={"message": "Database cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))