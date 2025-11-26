"""
Pinecone Index Setup and Embedding Script
Run this script once to create the Pinecone index and upload embeddings.
"""

import json
import os
import time
from typing import List, Dict
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Configuration
CHUNKS_FILE = "haas_mill_chunks.json"
EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_INDEX_NAME = "haas-mill-manual"
PINECONE_NAMESPACE = "operator-manual"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 100

def load_chunks(filepath: str) -> List[Dict]:
    """Load chunks from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def create_embeddings_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Create embeddings for a batch of texts."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def setup_pinecone_index(pc: Pinecone, index_name: str) -> None:
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(30)
    else:
        print(f"Index '{index_name}' already exists.")

def upload_embeddings(
    chunks: List[Dict],
    openai_client: OpenAI,
    pinecone_index,
    namespace: str,
    batch_size: int = BATCH_SIZE
) -> None:
    """Generate embeddings and upload to Pinecone."""
    
    print(f"Processing {len(chunks)} chunks...")
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        
        # Get texts for embedding
        texts = [chunk["text"] for chunk in batch]
        
        # Create embeddings
        embeddings = create_embeddings_batch(openai_client, texts)
        
        # Prepare vectors for upload
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"][:8000],  # Pinecone metadata limit
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"]
                }
            })
        
        # Upload to Pinecone
        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"Successfully uploaded {len(chunks)} vectors to Pinecone.")

def main():
    """Main setup function."""
    # Get API keys from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key or not pinecone_api_key:
        print("Error: Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export PINECONE_API_KEY='your-pinecone-key'")
        return
    
    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")
    
    # Initialize clients
    print("Initializing API clients...")
    openai_client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Setup Pinecone index
    setup_pinecone_index(pc, PINECONE_INDEX_NAME)
    
    # Get index reference
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Upload embeddings
    print("Generating and uploading embeddings...")
    upload_embeddings(chunks, openai_client, index, PINECONE_NAMESPACE)
    
    # Verify upload
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    main()
