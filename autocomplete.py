import os
from opensearchpy import OpenSearch
from typing import List
from pydantic import BaseModel
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products")

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)


class AutocompleteResponse(BaseModel):
    query: str
    total: int
    suggestions: List[str]


def search_autocomplete(
    query: str,
    fuzzy: bool = True,
    size: int = 10
) -> AutocompleteResponse:
    """
    Perform autocomplete search with fuzzy matching

    Args:
        query: Search query string
        fuzzy: Enable fuzzy matching (default: True)
        size: Number of results to return (default: 10)

    Returns:
        AutocompleteResponse with list of matching product titles
    """
    try:
        if fuzzy:
            search_body = {
                'query': {
                    'bool': {
                        'should': [
                            {
                                'match': {
                                    'title': {
                                        'query': query,
                                        'operator': 'and',
                                        'boost': 2
                                    }
                                }
                            },
                            {
                                'match': {
                                    'title': {
                                        'query': query,
                                        'fuzziness': 'AUTO',
                                        'prefix_length': 1
                                    }
                                }
                            }
                        ]
                    }
                },
                'size': size,
                '_source': ['title']
            }
        else:
            search_body = {
                'query': {
                    'match': {
                        'title': {
                            'query': query,
                            'operator': 'and'
                        }
                    }
                },
                'size': size,
                '_source': ['title']
            }

        response = client.search(index=INDEX_NAME, body=search_body)

        # Extract only the titles
        suggestions = [
            hit['_source']['title']
            for hit in response['hits']['hits']
        ]

        return AutocompleteResponse(
            query=query,
            total=response['hits']['total']['value'],
            suggestions=suggestions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenSearch query failed: {str(e)}")


def check_opensearch_health() -> dict:
    """
    Check OpenSearch cluster health

    Returns:
        dict with cluster status information
    """
    try:
        info = client.info()
        return {
            "status": "healthy",
            "cluster_name": info['cluster_name'],
            "version": info['version']['number']
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
