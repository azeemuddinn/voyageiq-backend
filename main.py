import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from database import supabase
from google import genai
from fastapi import UploadFile, File, Form
from pypdf import PdfReader
import io

load_dotenv()

app = FastAPI(title="VoyageIQ API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the official Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class DocumentIngest(BaseModel):
    title: str
    content: str
    source_type: str = "text"

@app.post("/ingest")
def ingest_document(doc: DocumentIngest):
    try:
        # 1. Insert parent document record
        res = supabase.table("documents").insert({
            "title": doc.title,
            "source_type": doc.source_type
        }).execute()
        
        if not res.data:
            raise HTTPException(status_code=400, detail="Failed to create document record.")
            
        doc_id = res.data[0]["id"]
        
        # 2. Chunk text by paragraphs
        chunks = [c.strip() for c in doc.content.split("\n\n") if c.strip()]
        
        if not chunks:
            return {"status": "success", "chunks_saved": 0}

        # 3. Generate vectors safely item-by-item to prevent list index mismatches
        for chunk in chunks:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk,
            )
            
            # Extract the vector values cleanly
            vector_values = response.embeddings[0].values
            
            supabase.table("document_chunks").insert({
                "document_id": doc_id,
                "content": chunk,
                "embedding": vector_values
            }).execute()
            
        return {"status": "success", "document_id": doc_id, "chunks_saved": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class ChatRequest(BaseModel):
    question: str   
    document_id: str | None = None

@app.post("/chat")
def chat_with_docs(req: ChatRequest):
    try:
        # 1. Embed the user's question
        q_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=req.question,
        )
        q_vector = q_response.embeddings[0].values

        # 2. Query Supabase
        match_res = supabase.rpc("match_document_chunks", {
            "query_embedding": q_vector,
            "match_threshold": 0.0, 
            "match_count": 4        
        }).execute()

        chunks = match_res.data
        
        # DEBUG: Print what chunks were retrieved in your FastAPI terminal
        print("--- DEBUG CHUNKS RETRIEVED ---")
        print(chunks)
        print("------------------------------")

        if req.document_id and chunks:
            chunks = [c for c in chunks if c["document_id"] == req.document_id]

        if not chunks:
            return {"answer": "I couldn't find any relevant information in your uploaded documents to answer this.", "sources_used": 0}

        context_text = "\n\n---\n\n".join([c["content"] for c in chunks])

        prompt = f"""You are VoyageIQ, a helpful AI travel assistant. Answer the user's question using ONLY the context provided below. If the answer cannot be found in the context, say "I cannot find that in your uploaded documents."

Context:
{context_text}

User Question: {req.question}
Answer:"""

        ai_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return {
            "answer": ai_response.text,
            "sources_used": len(chunks)
        }

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
     
@app.post("/ingest-pdf")
async def ingest_pdf(title: str = Form(...), file: UploadFile = File(...)):
    try:
        # 1. Read PDF bytes
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
        
        # 2. Extract text from all pages
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n\n"
                
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # 3. Insert parent document record
        res = supabase.table("documents").insert({
            "title": title,
            "source_type": "pdf"
        }).execute()
        
        if not res.data:
            raise HTTPException(status_code=400, detail="Failed to create document record.")
            
        doc_id = res.data[0]["id"]
        
        # 4. Chunk text and embed
        chunks = [c.strip() for c in extracted_text.split("\n\n") if c.strip()]
        
        for chunk in chunks:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk,
            )
            vector_values = response.embeddings[0].values
            
            supabase.table("document_chunks").insert({
                "document_id": doc_id,
                "content": chunk,
                "embedding": vector_values
            }).execute()
            
        return {"status": "success", "document_id": doc_id, "chunks_saved": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents():
    try:
        res = supabase.table("documents").select("id, title, created_at").order("created_at", desc=True).execute()
        return {"documents": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        