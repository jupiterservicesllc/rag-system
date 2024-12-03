# RAG System

## Overview
PDF document analysis and querying system using free and open-source components.

## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
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
```