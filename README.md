# 🌍 VoyageIQ — Backend (FastAPI RAG Service)

This is the Python/FastAPI backend repository for **VoyageIQ**, handling document parsing, vector embeddings, and strict RAG grounding to prevent AI hallucinations.

> 💻 **Looking for the UI?** Check out the [VoyageIQ Frontend Repository](https://github.com/azeemuddinn/voyageiq-frontend).

![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Backend](https://img.shields.io/badge/Backend-FastAPI-005571) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

---

## 🚀 Getting Started

Clone the repository and set up your local environment:

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server locally
uvicorn main:app --reload