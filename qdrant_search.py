from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from ollama_embeddings import embed_text_ollama
from image_embeddings import embed_image_marqo_from_base64, embed_image_marqo_from_url


class ProductSearch:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6336):
        """
        Initialize the product search with Qdrant client.

        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "products"

    def search_by_text(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Search products using text query with Ollama embeddings.

        Args:
            query_text: The search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional Qdrant filters to apply

        Returns:
            List of search results with scores and payloads
        """
        # Generate text embedding using Ollama
        query_vector = embed_text_ollama(query_text)

        # Search in Qdrant using query_points
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="vector_text",
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filters
        ).points

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return results

    def search_by_image_url(
        self,
        image_url: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Search products using image URL with Marqo embeddings.

        Args:
            image_url: URL of the image to search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional Qdrant filters to apply

        Returns:
            List of search results with scores and payloads
        """
        # Generate image embedding using Marqo
        query_vector = embed_image_marqo_from_url(image_url)

        # Search in Qdrant using query_points
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="vector_image",
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filters
        ).points

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return results

    def search_by_image_base64(
        self,
        base64_image: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Search products using base64 encoded image with Marqo embeddings.

        Args:
            base64_image: Base64 encoded image string
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional Qdrant filters to apply

        Returns:
            List of search results with scores and payloads
        """
        # Generate image embedding using Marqo
        query_vector = embed_image_marqo_from_base64(base64_image)

        # Search in Qdrant using query_points
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="vector_image",
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filters
        ).points

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return results

    def hybrid_search(
        self,
        query_text: Optional[str] = None,
        image_url: Optional[str] = None,
        base64_image: Optional[str] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining text and image queries.

        Args:
            query_text: Optional text query
            image_url: Optional image URL
            base64_image: Optional base64 encoded image
            text_weight: Weight for text search results (0-1)
            image_weight: Weight for image search results (0-1)
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional Qdrant filters to apply

        Returns:
            List of combined search results with weighted scores
        """
        results_map = {}

        # Text search
        if query_text:
            text_results = self.search_by_text(
                query_text=query_text,
                limit=limit * 2,  # Get more results for better merging
                score_threshold=score_threshold,
                filters=filters
            )
            for result in text_results:
                result_id = result["id"]
                weighted_score = result["score"] * text_weight
                if result_id in results_map:
                    results_map[result_id]["score"] += weighted_score
                else:
                    results_map[result_id] = {
                        "id": result_id,
                        "score": weighted_score,
                        "payload": result["payload"]
                    }

        # Image search
        if image_url:
            image_results = self.search_by_image_url(
                image_url=image_url,
                limit=limit * 2,
                score_threshold=score_threshold,
                filters=filters
            )
            for result in image_results:
                result_id = result["id"]
                weighted_score = result["score"] * image_weight
                if result_id in results_map:
                    results_map[result_id]["score"] += weighted_score
                else:
                    results_map[result_id] = {
                        "id": result_id,
                        "score": weighted_score,
                        "payload": result["payload"]
                    }

        elif base64_image:
            image_results = self.search_by_image_base64(
                base64_image=base64_image,
                limit=limit * 2,
                score_threshold=score_threshold,
                filters=filters
            )
            for result in image_results:
                result_id = result["id"]
                weighted_score = result["score"] * image_weight
                if result_id in results_map:
                    results_map[result_id]["score"] += weighted_score
                else:
                    results_map[result_id] = {
                        "id": result_id,
                        "score": weighted_score,
                        "payload": result["payload"]
                    }

        # Sort by combined score and limit results
        combined_results = sorted(
            results_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:limit]

        return combined_results


if __name__ == "__main__":
    import time

    # Initialize search
    search = ProductSearch()

    print("=== Text Search Example ===")
    start = time.time()
    text_results = search.search_by_text(
        query_text="FR01-CH" ,
        limit=9,
        filters = models.Filter(
    must=[
        models.FieldCondition(
            key="SKU",
            match=models.MatchValue(value="FR01-CH"),
        ),
    ],
    should=[],
    must_not=[],
)
    )
    end = time.time()
    print(f"Time taken: {end - start:.3f} seconds")
    print(f"Found {len(text_results)} results")
    for i, result in enumerate(text_results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Payload: {result['payload']}")

#     print("\n\n=== Image Search Example ===")
    # test_image_url = "https://lipkahome.com/cdn/shop/files/1_8d780b53-b507-4b2f-b80e-9d0b19fc2deb.jpg"
#     start = time.time()
#     image_results = search.search_by_image_url(
#         image_url=test_image_url,
#         limit=5,
#         filters = models.Filter(
#     must=[],
#     should=[],
#     must_not=[
#         models.FieldCondition(
#             key="Primary_Color",
#             match=models.MatchValue(value="Black"),
#         ),
#     ],
# )
#     )
#     end = time.time()
#     print(f"Time taken: {end - start:.3f} seconds")
#     print(f"Found {len(image_results)} results")
#     for i, result in enumerate(image_results, 1):
#         print(f"\n{i}. Score: {result['score']:.4f}")
#         print(f"   Payload: {result['payload']}")

    # print("\n\n=== Hybrid Search Example ===")
    # start = time.time()
    # hybrid_results = search.hybrid_search(
    #     query_text="riobel faucet",
    #     image_url=test_image_url,
    #     text_weight=0.6,
    #     image_weight=0.4,
    #     limit=5
    # )
    # end = time.time()
    # print(f"Time taken: {end - start:.3f} seconds")
    # print(f"Found {len(hybrid_results)} results")
    # for i, result in enumerate(hybrid_results, 1):
    #     print(f"\n{i}. Combined Score: {result['score']:.4f}")
    #     print(f"   Payload: {result['payload']}")
