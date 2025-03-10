from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .models import SearchQuery, SearchResult
from .utils import setup_clip_model, setup_pinecone, perform_hybrid_search
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Hybrid Search API", 
              description="API for hybrid search using CLIP and Pinecone")

# Initialize models and connections on startup
@app.on_event("startup")
async def startup_event():
    # Setup CLIP model
    app.state.clip_model = setup_clip_model()
    
    # Setup Pinecone
    app.state.pinecone_index = setup_pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name= utils.create_dlai_index_name('dl-ai') 
    )

@app.get("/")
def read_root():
    return {"message": "Hybrid Search API is running"}

@app.post("/search", response_model=list[SearchResult])
def search(query: SearchQuery):
    try:
        results = perform_hybrid_search(
            query=query.query,
            clip_model=app.state.clip_model,
            pinecone_index=app.state.pinecone_index,
            top_k=query.top_k
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 