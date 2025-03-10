# Extract relevant functions from your notebook
import torch
import clip
from pinecone import Pinecone

def setup_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return {"model": model, "preprocess": preprocess, "device": device}

def setup_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def perform_hybrid_search(query, clip_model, pinecone_index, bm25_encoder, top_k=5, alpha=0.5):
    """
    Perform a hybrid search using both dense and sparse vectors.

    Args:
        query (str): The search query.
        clip_model: The dense vector model (e.g., SentenceTransformer).
        pinecone_index: The Pinecone index to query.
        bm25_encoder: The BM25 encoder for sparse vector creation.
        top_k (int): Number of top results to return.
        alpha (float): Weighting factor for dense vs sparse (0 = sparse only, 1 = dense only).

    Returns:
        list: Top-k search results with metadata.
    """
    # Encode the query into dense and sparse vectors
    dense_vector = clip_model.encode(query).tolist()
    sparse_vector = bm25_encoder.encode_queries(query)

    # Scale the vectors for hybrid search
    hdense, hsparse = hybrid_scale(dense_vector, sparse_vector, alpha)

    # Query the Pinecone index
    results = pinecone_index.query(
        top_k=top_k,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )

    return results['matches']

def hybrid_scale(dense, sparse, alpha):
    """
    Scale dense and sparse vectors for hybrid search.

    Args:
        dense (list): Dense vector.
        sparse (dict): Sparse vector with 'indices' and 'values'.
        alpha (float): Weighting factor for dense vs sparse.

    Returns:
        tuple: Scaled dense and sparse vectors.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Scale sparse and dense vectors
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    
    return hdense, hsparse
    
    # Example placeholder implementation:
    model = clip_model["model"]
    preprocess = clip_model["preprocess"]
    device = clip_model["device"]
    
    # Encode the query using CLIP
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([query]).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convert to list for Pinecone
    query_embedding = text_features.cpu().numpy().tolist()[0]
    
    # Query Pinecone
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results
    formatted_results = []
    for match in results["matches"]:
        formatted_results.append({
            "id": match["id"],
            "score": float(match["score"]),
            "product_name": match["metadata"].get("productDisplayName", ""),
            "image_url": match["metadata"].get("image", None),
            "metadata": match["metadata"]
        })
    
    return formatted_results 