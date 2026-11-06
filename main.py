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
from sentence_transformers import CrossEncoder

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
            raise HTTPException(status_code=400, detail="Failed to create document rec   ord.")
            
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

# reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# @app.post("/chat")
# def chat_with_docs(req: ChatRequest):
#     try:
#         # 1. Embed the user's question for vector search
#         q_response = client.models.embed_content(
#             model="gemini-embedding-001",
#             contents=req.question,
#         )
#         q_vector = q_response.embeddings[0].values

#         # 2. Project 3: Hybrid Search RPC (combines vector similarity + keyword search)
#         # We fetch a larger candidate pool (e.g., 20) for the re-ranker to filter
#         match_res = supabase.rpc("hybrid_search_document_chunks", {
#             "query_text": req.question,
#             "query_embedding": q_vector,
#             "match_count": 20        
#         }).execute()

#         chunks = match_res.data
        
#         if req.document_id and chunks:
#             chunks = [c for c in chunks if c["document_id"] == req.document_id]

#         if not chunks:
#             return {"answer": "I couldn't find any relevant information in your uploaded documents to answer this.", "sources_used": 0}

#         # 3. Project 5: Cross-Encoder Re-ranking
#         # Pair the user question with each retrieved chunk text for deep semantic scoring
#         pairs = [[req.question, chunk["content"]] for chunk in chunks]
#         scores = reranker_model.predict(pairs)

#         # Attach the cross-encoder score to each chunk and sort descending
#         for i, chunk in enumerate(chunks):
#             chunk["relevance_score"] = float(scores[i])

#         chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

#         # Keep only the top 4 highest-scoring re-ranked chunks
#         top_chunks = chunks[:4]

#         # DEBUG: Inspect top re-ranked chunks
#         print("--- TOP RE-RANKED CHUNKS ---")
#         for c in top_chunks:
#             print(f"Score: {c['relevance_score']:.4f} | Content: {c['content'][:100]}...")
#         print("----------------------------")

#         context_text = "\n\n---\n\n".join([c["content"] for c in top_chunks])

#         prompt = f"""You are VoyageIQ, a helpful AI travel assistant. Answer the user's question using ONLY the context provided below. If the answer cannot be found in the context, say "I cannot find that in your uploaded documents."

# Context:
# {context_text}

# User Question: {req.question}
# Answer:"""

#         ai_response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt
#         )

#         return {
#             "answer": ai_response.text,
#             "sources_used": len(top_chunks)
#         }

#     except Exception as e:
#         print("ERROR:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))


app.post("/chat")
def chat_with_docs(req: ChatRequest):
    try:
        # 1. Generate embedding for user query
        embedding_response = client.models.embed_content(
            model="text-embedding-004",
            contents=req.question
        )
        query_embedding = embedding_response.embeddings[0].values

        # 2. Call Supabase Hybrid Search RPC (Project 3: Dense + Sparse RRF)
        rpc_response = supabase.rpc(
            "hybrid_search_document_chunks",
            {
                "query_text": req.question,
                "query_embedding": query_embedding,
                "match_count": req.match_count * 2  # Fetch a wider candidate pool
            }
        ).execute()

        chunks = rpc_response.data
        if not chunks:
            return {"answer": "I couldn't find any relevant information in the documents."}

        # =========================================================================
        # [ADVANCED - Project 5] Local Cross-Encoder Re-ranking Pipeline
        # To enable: uncomment lines below (Requires paid server with >512MB RAM)
        # =========================================================================
        # pairs = [[req.question, chunk["content"]] for chunk in chunks]
        # scores = reranker_model.predict(pairs)
        # for i, chunk in enumerate(chunks):
        #     chunk["relevance_score"] = float(scores[i])
        # chunks = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)
        # selected_chunks = chunks[:req.match_count]
        # =========================================================================

        # Stable fallback: use top hybrid search results directly
        selected_chunks = chunks[:req.match_count]

        # 3. Format context for Gemini
        context_text = "\n\n".join([chunk["content"] for chunk in selected_chunks])
        prompt = f"Use the following context to answer the user's question accurately.\n\nContext:\n{context_text}\n\nQuestion: {req.question}"

        # 4. Generate response via Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources_used": len(selected_chunks)
        }

    except Exception as e:
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