import os
# from openai import OpenAI
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import uuid
import time
from dotenv import load_dotenv
load_dotenv()

# Initialize clients
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(host="localhost", port=6336)

COLLECTION_NAME = "unique_companies"
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


def load_companies(file_path: str) -> List[str]:
    """Load companies from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]
    return companies


def store_companies(companies: List[str]):
    """Store companies with embeddings in Qdrant."""
    points = []

    for idx, company in enumerate(companies):
        print(f"Processing {idx + 1}/{len(companies)}: {company}")

        # Get embedding
        embedding = get_embedding(company)

        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"company_name": company, "index": idx}
        )
        points.append(point)

    # Upload to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"\nStored {len(points)} companies in Qdrant.")


def search_companies(query: str, limit: int = 10) -> List[Dict]:
    """Search for similar companies using semantic search."""
    # Get query embedding
    query_embedding = get_embedding(query)

    start = time.time()
    # Search in Qdrant using query_points (updated API)
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit
    )
    end = time.time()
    print(f"Search took {end - start:.2f} seconds.")
    # print(search_results)

    start_result = time.time()
    # Format results
    results = []
    for result in search_results.points:
        results.append((
            result.payload["company_name"],
            f"Similarity Score: {result.score}"
        ))
    end_result = time.time()
    print(f"Result formatting took {end_result - start_result:.2f} seconds.")
    return results


def main():
    """Main function to store companies."""
    # # Create collection
    # create_collection()

    # # # Load companies
    # companies = load_companies("companies.txt")
    # print(f"Loaded {len(companies)} companies from file.")

    # # # Store companies
    # store_companies(companies)

    # Example similarity search
    print("\n" + "="*50)
    print("Example similarity search:")
    print("="*50)
    query = "show me something in braizon and quabress"
    start = time.time()
    results = search_companies(query, limit=10)
    end = time.time()
    print(f"Search completed in {end - start:.2f} seconds.")
    print(results)
    print(f"\nQuery: '{query}'")
    print("\nTop matches:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['company_name']} (score: {result['score']:.4f})")


if __name__ == "__main__":
    main()
