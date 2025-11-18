from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(host='localhost', port=6336)

# Get all collections
collections = client.get_collections()

# Print collection names
for collection in collections.collections:
    print(collection.name)