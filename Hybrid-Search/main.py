from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import warnings
from datasets import load_dataset
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from DLAIUtils import Utils
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
import os
import nltk
import base64
from PIL import Image
import io

# Suppress warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Mount static files for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Function to setup Pinecone
def setup_pinecone():
    utils = Utils()
    PINECONE_API_KEY = utils.get_pinecone_api_key()
    INDEX_NAME = utils.create_dlai_index_name('dl-ai')
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(INDEX_NAME)
    pinecone.create_index(
        INDEX_NAME,
        dimension=512,
        metric="dotproduct",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    return pinecone.Index(INDEX_NAME)

# Function to load dataset
def load_fashion_dataset():
    fashion = load_dataset("ashraq/fashion-product-images-small", split="train")
    return fashion

# Function to create sparse vector using BM25
def create_sparse_vector(metadata):
    bm25 = BM25Encoder()
    nltk.download('punkt')
    bm25.fit(metadata['productDisplayName'])
    return bm25

# Function to create dense vector using CLIP
def create_dense_vector(text, device):
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    dense_vec = model.encode([text])
    return dense_vec

# Function to convert image to base64 for HTML display
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/process", response_class=HTMLResponse)
def process_data(request: Request):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup Pinecone
    index = setup_pinecone()

    # Load dataset
    fashion = load_fashion_dataset()
    images = fashion['image']
    metadata = fashion.remove_columns('image').to_pandas()

    # Create vectors
    bm25 = create_sparse_vector(metadata)
    dense_vec = create_dense_vector(metadata['productDisplayName'][0], device)

    # Get the first image and convert it to base64
    image = images[0]
    image_base64 = image_to_base64(image)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "product_name": metadata['productDisplayName'][0],
        "dense_vector_shape": dense_vec.shape,
        "image_base64": image_base64
    })

@app.get("/search")
async def search_products(query: str = Query(..., description="Search query")):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load dataset
        print("Loading dataset...")
        fashion = load_fashion_dataset()
        images = fashion['image']
        metadata = fashion.remove_columns('image').to_pandas()

        # Create dense vector for the query
        print("Creating dense vector for the query...")
        dense_query_vec = create_dense_vector(query, device)

        # Create sparse vector for the query
        print("Creating sparse vector for the query...")
        bm25 = create_sparse_vector(metadata)
        sparse_query_vec = bm25.encode_queries(query)

        # Hybrid search (combine dense and sparse vectors)
        alpha = 0.5  # Weight for dense vs. sparse vectors
        hybrid_query_vec = {
            "dense": (alpha * dense_query_vec[0]).tolist(),  # Convert to list for Pinecone
            "sparse": {
                "indices": sparse_query_vec["indices"],
                "values": [(1 - alpha) * value for value in sparse_query_vec["values"]]
            }
        }

        # Perform search in Pinecone
        print("Performing search in Pinecone...")
        index = setup_pinecone()
        results = index.query(
            vector=hybrid_query_vec,
            top_k=5,  # Return top 5 results
            include_metadata=True
        )

        # Prepare results for the frontend
        print("Preparing results for the frontend...")
        response = []
        for match in results['matches']:
            product_id = match['id']
            if int(product_id) >= len(metadata) or int(product_id) >= len(images):
                print(f"Invalid product_id: {product_id}")
                continue

            product_metadata = metadata.iloc[int(product_id)]
            image = images[int(product_id)]
            image_base64 = image_to_base64(image)

            if image_base64 is None:
                continue

            response.append({
                "product_name": product_metadata['productDisplayName'],
                "dense_vector_shape": match['values'].shape,
                "image_base64": image_base64
            })

        return JSONResponse(content=response)
    except Exception as e:
        print(f"Error in /search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)