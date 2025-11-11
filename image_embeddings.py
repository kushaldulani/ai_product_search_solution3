from typing import List
import base64
from io import BytesIO

import requests
from PIL import Image
import torch
import open_clip

_MODEL = None
_PREPROCESS = None
import time

def _load_model() -> None:
    global _MODEL, _PREPROCESS
    if _MODEL is None or _PREPROCESS is None:
        model_name = "hf-hub:Marqo/marqo-ecommerce-embeddings-L"
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        _MODEL = model.eval()
        _PREPROCESS = preprocess_val


def embed_image_marqo_from_url(image_url: str, timeout: int = 60) -> List[float]:
    _load_model()
    response = requests.get(image_url, timeout=timeout, stream=True)
    response.raise_for_status()
    image = Image.open(response.raw).convert("RGB")
    image_tensor = _PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        image_features = _MODEL.encode_image(image_tensor, normalize=True)
    vector = image_features.squeeze(0).cpu().tolist()
    return vector


def embed_image_marqo_from_base64(base64_string: str) -> List[float]:
    """
    Generate image embeddings from a base64 encoded image string.

    Args:
        base64_string: Base64 encoded image string (with or without data URI prefix)

    Returns:
        List of embedding values
    """
    _load_model()

    # Remove data URI prefix if present (e.g., "data:image/png;base64,")
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Create tensor and generate embeddings
    image_tensor = _PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        image_features = _MODEL.encode_image(image_tensor, normalize=True)
    vector = image_features.squeeze(0).cpu().tolist()
    return vector



if __name__ == "__main__":
    # Test embedding from URL
    print("=== Testing embed_image_marqo_from_url ===")
    test_image_url = "https://cdn.shopify.com/s/files/1/0747/8014/7988/files/PSX01-MB.jpg?v=1752001464"
    # start = time.time()
    # embedding = embed_image_marqo_from_url(test_image_url)
    # end = time.time()
    # print(f"Time taken: {end - start} seconds")
    # print(f"Embedding length: {len(embedding)}")
    # print(f"First 5 values: {embedding[:5]}")
    # print(embedding)

    # Test embedding from base64
    print("\n=== Testing embed_image_marqo_from_base64 ===")
    # Convert the same image to base64 for testing
    import requests
    from io import BytesIO
    resp = requests.get(test_image_url)
    img_bytes = resp.content
    base64_string = base64.b64encode(img_bytes).decode('utf-8')

    start = time.time()
    embedding_base64 = embed_image_marqo_from_base64(base64_string)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(f"Embedding length: {len(embedding_base64)}")
    print(f"First 5 values: {embedding_base64[:5]}")

    # Verify embeddings are similar (should be identical for same image)
    # print("\n=== Comparing embeddings ===")
    # print(f"Embeddings match: {embedding == embedding_base64}")