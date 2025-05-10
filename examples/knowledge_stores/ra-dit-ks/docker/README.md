# Running with Docker

To run the Qdrant Atlas Subset (i.e., en-wiki Dec 2018) Vector DB:

```bash
# Pull the image
docker pull your-org/qdrant-atlas:latest

# Run the container
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage your-org/qdrant-atlas:latest
