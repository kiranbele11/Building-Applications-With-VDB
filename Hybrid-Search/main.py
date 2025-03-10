from fastapi import FastAPI, HTTPException
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

# Suppress warnings
warnings.filterwarnings('ignore')

app = FastAPI()

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
def create_dense_vector(metadata, device):
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    dense_vec = model.encode([metadata['productDisplayName'][0]])
    return dense_vec

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fashion Product Search API!"}

@app.get("/process")
def process_data():
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
    dense_vec = create_dense_vector(metadata, device)

    return {
        "product_name": metadata['productDisplayName'][0],
        "dense_vector_shape": dense_vec.shape
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)