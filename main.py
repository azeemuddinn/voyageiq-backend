import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from database import supabase
from google import genai

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
        # 3. Chunk text by paragraphs
        chunks = [c.strip() for c in doc.content.split("\n\n") if c.strip()]
        
        if not chunks:
            return {"status": "success", "chunks_saved": 0}

        # 3. Generate real vector embeddings using Gemini
        # We use gemini-embedding-2 and request 768 dimensions
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunks,
        )
        
        # 4. Save each chunk along with its vector array into Supabase
        for i, chunk in enumerate(chunks):
            vector_values = response.embeddings[i].values
            
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

@app.post("/chat")
def chat_with_docs(req: ChatRequest):
    try:
        # 1. Embed the user's question
        q_response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=req.question,
        )
        q_vector = q_response.embeddings[0].values

        # 2. Query Supabase for relevant chunks using the function we just created
        match_res = supabase.rpc("match_document_chunks", {
            "query_embedding": q_vector,
            "match_threshold": 0.3, # Adjust confidence threshold as needed
            "match_count": 4        # Bring top 4 matching chunks
        }).execute()

        chunks = match_res.data
        
        if not chunks:
            return {"answer": "I couldn't find any relevant information in your uploaded documents to answer this."}

        # 3. Build context from retrieved chunks
        context_text = "\n\n---\n\n".join([c["content"] for c in chunks])

        # 4. Generate the final answer using Gemini Flash or Pro
        prompt = f"""You are VoyageIQ, a helpful AI travel assistant. Answer the user's question using ONLY the context provided below. If the answer cannot be found in the context, say "I cannot find that in your uploaded documents."

Context:
{context_text}

User Question: {req.question}
Answer:"""

        ai_response = client.models.generate_content(
            model="gemini-2.5-flash", # Or gemini-1.5-flash
            contents=prompt
        )

        return {
            "answer": ai_response.text,
            "sources_used": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))