import os
from pathlib import Path

def generate_code_contents():
    # Define all file contents
    contents = {
        "requirements.txt": """# Core Web Framework
fastapi==0.109.0
uvicorn==0.24.0
python-multipart==0.0.9
jinja2==3.1.3

# PDF Processing
pdfplumber==0.10.3

# Vector Database
chromadb==0.4.22

# Embedding Model
sentence-transformers==2.5.1

# LLM
llama-cpp-python==0.2.27

# Optional: CPU-only PyTorch
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.1.2+cpu""",

        "README.md": """# RAG System

## Overview
PDF document analysis and querying system using free and open-source components.

## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download LLAMA model:
- Get a GGUF model from https://huggingface.co/TheBloke
- Place it in the `models` directory

4. Run:
```bash
uvicorn app.main:app --reload
```""",

        "app/main.py": """import os
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        temp_path = TEMP_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        text_chunks = pdf_processor.extract_text(str(temp_path))
        embeddings = embedding_model.generate_embeddings(text_chunks)
        vector_db.add_documents(text_chunks, embeddings)

        temp_path.unlink()
        return {"status": "success", "message": f"Processed {file.filename}"}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: str):
    try:
        query_embedding = embedding_model.generate_query_embedding(query)
        context_docs = vector_db.search(query_embedding)
        response = query_model.generate_response(query, context_docs)
        return {"response": response}

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-database")
async def clear_database():
    try:
        vector_db.reset()
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        logger.error(f"Database reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))""",

        "app/database.py": """import chromadb
from typing import List
from pathlib import Path

class VectorDatabase:
    def __init__(self, persist_directory: str = "db"):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="document_collection"
        )

    def add_documents(self, documents: List[str], embeddings: List[List[float]]):
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

    def reset(self):
        self.client.delete_collection(name="document_collection")
        self.collection = self.client.get_or_create_collection(
            name="document_collection"
        )""",

        "app/models.py": """from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from typing import List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents: List[str]) -> List[List[float]]:
        return [embedding.tolist() for embedding in self.model.encode(documents)]

    def generate_query_embedding(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()

class QueryModel:
    def __init__(self, model_path: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Please download it first.")
            
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4
        )

    def generate_response(self, query: str, context_docs: List[str]) -> str:
        context = "\\n".join(context_docs)
        prompt = f\"\"\"Context information is below.
---------------------
{context}
---------------------
Given the context information, answer the following question:
{query}

If the answer cannot be found in the context, say "I cannot answer this based on the provided context."
Answer:\"\"\"

        response = self.llm(
            prompt,
            max_tokens=500,
            stop=["Question:", "Context:"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()""",

        "app/pdf_processor.py": """import pdfplumber
from typing import List
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text(self, pdf_path: str) -> List[str]:
        try:
            full_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    full_text += page.extract_text() or ""

            # Clean text
            full_text = full_text.replace('\\n', ' ').strip()
            
            # Create chunks
            chunks = []
            for i in range(0, len(full_text), self.chunk_size - self.overlap):
                chunk = full_text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise""",

        "app/templates/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>Document Query System</h1>
        
        <div class="upload-section">
            <h2>Upload PDF</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="pdfFile" accept=".pdf" required>
                <button type="submit">Upload</button>
            </form>
            <div id="uploadStatus"></div>
        </div>

        <div class="query-section">
            <h2>Query Documents</h2>
            <textarea id="queryInput" placeholder="Enter your question..." rows="3"></textarea>
            <button onclick="queryDocuments()">Ask Question</button>
            <div id="queryResponse" class="response-box"></div>
        </div>

        <div class="database-section">
            <h2>Database Management</h2>
            <button onclick="clearDatabase()" class="danger-button">Clear Database</button>
        </div>
    </div>

    <script>
        async function uploadPDF(event) {
            event.preventDefault();
            const fileInput = document.getElementById('pdfFile');
            const uploadStatus = document.getElementById('uploadStatus');
            
            if (!fileInput.files.length) {
                uploadStatus.textContent = 'Please select a file';
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                uploadStatus.textContent = 'Uploading...';
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                uploadStatus.textContent = result.message;
            } catch (error) {
                uploadStatus.textContent = 'Upload failed: ' + error.message;
            }
        }

        async function queryDocuments() {
            const queryInput = document.getElementById('queryInput');
            const queryResponse = document.getElementById('queryResponse');
            
            if (!queryInput.value.trim()) {
                queryResponse.textContent = 'Please enter a question';
                return;
            }

            try {
                queryResponse.textContent = 'Processing query...';
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: queryInput.value })
                });
                const result = await response.json();
                queryResponse.textContent = result.response;
            } catch (error) {
                queryResponse.textContent = 'Query failed: ' + error.message;
            }
        }

        async function clearDatabase() {
            if (!confirm('Are you sure you want to clear the database? This cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch('/clear-database', { method: 'POST' });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Failed to clear database: ' + error.message);
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', uploadPDF);
    </script>
</body>
</html>""",

        "app/static/style.css": """body {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f0f2f5;
    color: #333;
}

.container {
    max-width: 800px;
    margin: auto;
    background: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

h1, h2 {
    color: #1a1a1a;
    margin-bottom: 20px;
}

h1 {
    text-align: center;
    font-size: 2.2em;
}

h2 {
    font-size: 1.5em;
    border-bottom: 2px solid #eee;
    padding-bottom: 10px;
}

input[type="file"], 
textarea {
    width: 100%;
    padding: 12px;
    margin: 10px 0;
    border: 2px solid #ddd;
    border-radius: 6px;
    font-size: 16px;
    transition: border-color 0.3s;
}

textarea {
    resize: vertical;
    min-height: 100px;
}

input[type="file"]:focus, 
textarea:focus {
    border-color: #007bff;
    outline: none;
}

button {
    background-color: #007bff;
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #0056b3;
}

.danger-button {
    background-color: #dc3545;
}

.danger-button:hover {
    background-color: #c82333;
}

.upload-section, .query-section, .database-section {
    margin-bottom: 30px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}

#uploadStatus, #queryResponse {
    margin-top: 15px;
    padding: 15px;
    border-radius: 6px;
    background: #fff;
    border: 1px solid #ddd;
}

.response-box {
    white-space: pre-wrap;
    line-height: 1.5;
}""",

        "app/__init__.py": ""  # Empty file
    }

    # Write all files
    for file_path, content in contents.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Writing {file_path}")
        path.write_text(content)

    print("\nAll files generated successfully!")
    print("\nNext steps:")
    print("1. Create virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("\n2. Install requirements:")
    print("   pip install -r requirements.txt")
    print("\n3. Download a GGUF model:")
    print("   - Visit: https://huggingface.co/TheBloke")
    print("   - Download a GGUF model (e.g., TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf)")
    print("   - Place it in the 'models' directory")
    print("\n4. Run the application:")
    print("   uvicorn app.main:app --reload")
    print("\n5. Open in browser:")
    print("   http://localhost:8000")

if __name__ == "__main__":
    try:
        generate_code_contents()
    except Exception as e:
        print(f"Error: {e}")