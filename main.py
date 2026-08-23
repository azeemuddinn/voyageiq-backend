from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import supabase

app = FastAPI(title="VoyageIQ API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentIngest(BaseModel):
    title: str
    content: str
    source_type: str = "text"

@app.post("/ingest")
def ingest_document(doc: DocumentIngest):
    try:
        # 1. Insert into 'documents' parent table
        res = supabase.table("documents").insert({
            "title": doc.title,
            "source_type": doc.source_type
        }).execute()
        
        if not res.data:
            raise HTTPException(status_code=400, detail="Failed to create document record.")
            
        doc_id = res.data[0]["id"]
        
        # 2. Simple text chunking (split by paragraphs or periods for MVP)
        chunks = [c.strip() for c in doc.content.split("\n\n") if c.strip()]
        
        # 3. Save chunks (For now we save content, next we hook up the embedding generator)
        for chunk in chunks:
            supabase.table("document_chunks").insert({
                "document_id": doc_id,
                "content": chunk,
                #embedding: [0.0, ...] -> We will generate real vectors next!
            }).execute()
            
        return {"status": "success", "document_id": doc_id, "chunks_saved": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))