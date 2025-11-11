from typing import List
import ollama


def embed_text_ollama(text: str, model: str = "mxbai-embed-large") -> List[float]:

    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


if __name__ == "__main__":
    import time

    # Test embedding
    test_text = "Modern minimalist furniture with clean lines and neutral colors"

    print("=== Testing embed_text_ollama ===")
    start = time.time()
    embedding = embed_text_ollama(test_text)
    end = time.time()

    print(f"Text: {test_text}")
    print(f"Time taken: {end - start:.3f} seconds")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
