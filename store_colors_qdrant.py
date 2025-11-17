import os
import json
# from openai import OpenAI
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Set
import uuid
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(host="localhost", port=6336)

COLLECTION_NAME = "unique_colors"
# EMBEDDING_MODEL = "text-embedding-3-large"
# VECTOR_SIZE = 3072  # text-embedding-3-large dimension
EMBEDDING_MODEL = "mxbai-embed-large"
VECTOR_SIZE = 1024  # mxbai-embed-large dimension


def create_collection():
    """Create Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")


def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    # OpenAI version (commented out)
    # response = client.embeddings.create(
    #     model=EMBEDDING_MODEL,
    #     input=text
    # )
    # return response.data[0].embedding

    # Ollama version
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def load_colors_from_file(file_path: str = "data/colors.txt") -> List[str]:
    """Load colors from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        colors = [line.strip() for line in f if line.strip()]
    return colors


def store_colors(colors: List[str]):
    """Store colors with embeddings in Qdrant."""
    points = []
    colors_list = sorted(colors)  # Sort for consistent ordering

    for idx, color in enumerate(colors_list):
        print(f"Processing {idx + 1}/{len(colors_list)}: {color}")

        # Get embedding
        embedding = get_embedding(color)

        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"color_name": color, "index": idx}
        )
        points.append(point)

    # Upload to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"\nStored {len(points)} unique colors in Qdrant.")


def search_colors(query: str, limit: int = 10) -> List[Dict]:
    """Search for similar colors using semantic search."""
    # Get query embedding
    query_embedding = get_embedding(query)

    # Search in Qdrant using query_points (updated API)
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit
    )

    # Format results
    results = []
    for result in search_results.points:
        results.append((
            result.payload["color_name"],
            f"Similarity Score: {result.score}"
        ))

    return results


def main():
    """Main function to store colors."""
    # # Create collection
    create_collection()

    # Load colors from file
    colors = load_colors_from_file("data/colors.txt")
    print(f"Loaded {len(colors)} colors from colors.txt\n")

    # Store colors
    store_colors(colors)

    # Example similarity search
    print("\n" + "="*50)
    print("Example color similarity search:")
    print("="*50)

    test_queries = ["I don't want something in dark grey"]

    for query in test_queries:
        results = search_colors(query, limit=10)
        print(f"\nQuery: '{query}'")
        print("Top matches:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result[0]} ({result[1]})")


if __name__ == "__main__":
    main()
