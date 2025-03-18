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
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Function to query Pinecone index
def query_index(index, query_text, bm25, dense_vec, top_k=5):
    # Encode the query text into a sparse vector using BM25
    sparse_vec = bm25.encode_queries([query_text])
    
    # Convert NumPy arrays to lists for serialization
    dense_vector_list = dense_vec[0].tolist() if hasattr(dense_vec[0], 'tolist') else dense_vec[0]
    sparse_vector_dict = {
        'indices': sparse_vec[0]['indices'],
        'values': sparse_vec[0]['values'].tolist() if hasattr(sparse_vec[0]['values'], 'tolist') else sparse_vec[0]['values']
    }
    
    # Query the Pinecone index
    results = index.query(
        vector=dense_vector_list,
        sparse_vector=sparse_vector_dict,
        top_k=top_k,
        include_metadata=True
    )
    return results

# Main function
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.title("Fashion Product Search")

    # Setup Pinecone
    index = setup_pinecone()

    # Load dataset
    fashion = load_fashion_dataset()
    metadata = fashion.remove_columns('image').to_pandas()

    # Create BM25 encoder
    bm25 = create_sparse_vector(metadata)

    # User input
    query = st.text_input("Enter your search query:")

    if query:
        # Create dense vector for the query
        dense_vec = create_dense_vector(query, device)

        # Query the index
        results = query_index(index, query, bm25, dense_vec)

        # Display results
        st.write("## Search Results:")
        
        # Create columns for displaying images side by side
        cols = st.columns(min(5, len(results['matches'])))
        
        for i, match in enumerate(results['matches']):
            product_id = match['id']
            # Find the image in the original dataset
            product_idx = int(product_id) if product_id.isdigit() else None
            
            with cols[i % len(cols)]:
                st.write(f"**{match['metadata']['productDisplayName']}**")
                st.write(f"Score: {match['score']:.4f}")
                
                # Display image if available
                if product_idx is not None and product_idx < len(fashion):
                    try:
                        image = fashion[product_idx]['image']
                        st.image(image, caption=match['metadata']['productDisplayName'], use_column_width=True)
                    except Exception as e:
                        st.write(f"Could not display image: {e}")
                st.write("---")

if __name__ == "__main__":
    main()