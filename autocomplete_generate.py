#!/usr/bin/env python3
"""
OpenSearch Autocomplete Index Generator

This script:
1. Reads product titles from data/product_titles.txt
2. Creates an OpenSearch index optimized for autocomplete
3. Bulk indexes all product titles
4. Provides progress tracking and error handling
"""

import os
import sys
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products")
DATA_FILE = "./data/product_titles.txt"

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)


# Index settings optimized for autocomplete
INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "autocomplete_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"]
                },
                "search_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "autocomplete_analyzer",
                "search_analyzer": "search_analyzer",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            }
        }
    }
}


def check_connection() -> bool:
    """Check if OpenSearch is accessible"""
    try:
        info = client.info()
        print(f"✓ Connected to OpenSearch cluster: {info['cluster_name']}")
        print(f"  Version: {info['version']['number']}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to OpenSearch: {e}")
        print(f"  Make sure OpenSearch is running on {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
        return False


def create_index() -> bool:
    """Create the autocomplete index with optimized settings"""
    try:
        # Delete existing index if it exists
        if client.indices.exists(index=INDEX_NAME):
            print(f"⚠ Index '{INDEX_NAME}' already exists. Deleting...")
            client.indices.delete(index=INDEX_NAME)
            print(f"✓ Deleted existing index")

        # Create new index
        client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
        print(f"✓ Created index '{INDEX_NAME}' with autocomplete settings")
        return True

    except RequestError as e:
        print(f"✗ Failed to create index: {e.error}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error creating index: {e}")
        return False


def read_product_titles(file_path: str) -> List[str]:
    """Read product titles from file and remove duplicates"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            titles = [line.strip() for line in f if line.strip()]

        # Remove duplicates while preserving order
        unique_titles = list(dict.fromkeys(titles))

        print(f"✓ Read {len(titles)} titles from {file_path}")
        print(f"  Unique titles: {len(unique_titles)}")

        return unique_titles

    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        sys.exit(1)


def generate_documents(titles: List[str]) -> List[Dict]:
    """Generate documents for bulk indexing"""
    documents = []
    for i, title in enumerate(titles, start=1):
        documents.append({
            "_index": INDEX_NAME,
            "_id": i,
            "_source": {
                "title": title
            }
        })
    return documents


def bulk_index(documents: List[Dict]) -> bool:
    """Bulk index documents with progress tracking"""
    try:
        print(f"\n📝 Indexing {len(documents)} documents...")

        # Use helpers.bulk for efficient bulk indexing
        success, failed = helpers.bulk(
            client,
            documents,
            chunk_size=500,
            request_timeout=30,
            raise_on_error=False,
            stats_only=False
        )

        print(f"✓ Successfully indexed {success} documents")

        if failed:
            print(f"⚠ Failed to index {len(failed)} documents")
            for item in failed[:5]:  # Show first 5 failures
                print(f"  - {item}")

        # Refresh index to make documents searchable immediately
        client.indices.refresh(index=INDEX_NAME)
        print(f"✓ Index refreshed")

        return True

    except Exception as e:
        print(f"✗ Bulk indexing failed: {e}")
        return False


def verify_index() -> None:
    """Verify the index was created successfully"""
    try:
        count = client.count(index=INDEX_NAME)
        total = count['count']
        print(f"\n📊 Index Statistics:")
        print(f"  Total documents: {total}")

        # Test search
        test_query = {
            "query": {
                "match": {
                    "title": "faucet"
                }
            },
            "size": 5
        }

        response = client.search(index=INDEX_NAME, body=test_query)
        hits = len(response['hits']['hits'])

        print(f"\n🔍 Test Search (query='faucet'):")
        print(f"  Found {response['hits']['total']['value']} matches")
        print(f"  Top {hits} results:")
        for hit in response['hits']['hits'][:5]:
            print(f"    - {hit['_source']['title']}")

    except Exception as e:
        print(f"⚠ Verification failed: {e}")


def main():
    """Main execution flow"""
    print("=" * 60)
    print("OpenSearch Autocomplete Index Generator")
    print("=" * 60)
    print()

    # Step 1: Check connection
    if not check_connection():
        sys.exit(1)

    print()

    # Step 2: Create index
    if not create_index():
        sys.exit(1)

    print()

    # Step 3: Read product titles
    titles = read_product_titles(DATA_FILE)

    print()

    # Step 4: Generate documents
    print("📦 Preparing documents for indexing...")
    documents = generate_documents(titles)
    print(f"✓ Generated {len(documents)} documents")

    # Step 5: Bulk index
    if not bulk_index(documents):
        sys.exit(1)

    # Step 6: Verify
    verify_index()

    print()
    print("=" * 60)
    print("✅ Index generation completed successfully!")
    print("=" * 60)
    print()
    print(f"Index name: {INDEX_NAME}")
    print(f"OpenSearch URL: http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
    print()
    print("You can now use the /autocomplete endpoint in your API!")


if __name__ == "__main__":
    main()
