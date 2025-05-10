# Example: Wikipedia Dec 2021 Knowledge Store

A Docker image providing a pre-built Qdrant vector database with an Atlas corpus knowledge store for retrieval-augmented applications.

⚠️ **Note:** This Docker image is approximately 3.6GB in size due to the included Python environment and ML libraries. Ensure you have sufficient disk space and bandwidth when pulling the image.

## Quick Start

```bash
# Pull the image
docker pull vectorinstitute/qdrant-atlas-dec-wiki-2021:latest

# Run the container with basic settings and gpu acceleration
docker run -d \
  --name qdrant-vector-db \
  --gpus all \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  -e TINY_SAMPLE=true \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

On first run, the container will take some time to initialize as it sets up the knowledge store. The container will show (health: starting) status during initialization and will change to (healthy) when ready to use.

Access the dashboard: <http://localhost:6333/dashboard>

## Features

- Pre-built Qdrant vector database
- Built-in Atlas corpus knowledge store
- GPU acceleration support
- Configurable retrieval models
- Support for tiny samples for testing
- Health monitoring

## Installation

### Prerequisites

- Docker installed on your system
- At least 5GB of free disk space for the image
- At least 4GB of RAM (8GB+ recommended)
- (Optional) NVIDIA GPU with drivers for acceleration
- (Optional) NVIDIA Docker for GPU support

### Pulling the Image

```bash
# This will download approximately 3.6GB of data
docker pull vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

The initial pull may take some time depending on your internet connection speed.

## Running Options

### Basic Usage

```bash
docker run -d \
  --name qdrant-vector-db \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

This will:

- Start a container named `qdrant-vector-db`
- Expose the HTTP API on port 6333 and gRPC API on port 6334
- Mount a volume named `qdrant_data` for persistent storage
- On first run, build a knowledge store with the default settings

### Using GPU Acceleration

```bash
docker run -d \
  --name qdrant-vector-db \
  --gpus all \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

### Testing with a Small Sample

For development or testing, you can use a tiny sample data file instead of downloading the full corpus:

```bash
docker run -d \
  --name qdrant-vector-db-test \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  -e TINY_SAMPLE=true \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

## Configuration Options

The container supports several environment variables for customization:

```bash
docker run -d \
  --name qdrant-vector-db \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  -e MODEL_NAME="" \
  -e QUERY_MODEL_NAME="nthakur/dragon-plus-query-encoder" \
  -e CONTEXT_MODEL_NAME="nthakur/dragon-plus-context-encoder" \
  -e BATCH_SIZE=5000 \
  -e CLEAR_FIRST=True \
  -e CORPUS="enwiki-dec2021" \
  -e TINY_SAMPLE=false \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Single model name to use for both query and context encoding | "" (empty) |
| `QUERY_MODEL_NAME` | Model for query encoding | "nthakur/dragon-plus-query-encoder" |
| `CONTEXT_MODEL_NAME` | Model for context encoding | "nthakur/dragon-plus-context-encoder" |
| `BATCH_SIZE` | Batch size for processing data | 5000 |
| `CLEAR_FIRST` | Whether to clear existing collections | True |
| `CORPUS` | Which Atlas corpus to use | "enwiki-dec2021" |
| `TINY_SAMPLE` | Use a tiny sample for testing | false |
| `SKIP_DOWNLOAD` | Skip downloading corpus (use pre-mounted data) | false |
| `FILENAME` | Name of file to use from corpus | "text-list-100-sec.jsonl" |

## Using the Knowledge Store

Once the container is `healthy` status, then we can use

### Using with fed-rag

```python
from fed_rag.retriever.knowledge_store import QdrantKnowledgeStore
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)

# build retriever for encoding queries
retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
    load_model_at_init=False,
)

# Connect to the containerized knowledge store
knowledge_store = QdrantKnowledgeStore(
    collection_name="nthakur.dragon-plus-context-encoder",
    host="localhost",
    port=6333,
)

# Retrieve documents
query = "What is the history of marine biology?"
query_emb = retriever.encode_query(query).tolist()

results = knowledge_store.retrieve(query_emb=query_emb, top_k=3)
for node in results:
    print(f"Score: {node.score}, Content: {str(node.node)}")
```

### REST API

Query the collection:

```bash
curl -X POST 'http://localhost:6333/collections/nthakur.dragon-plus-context-encoder/points/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, ..., 0.8],
    "limit": 3
  }'
```

List collections:

```bash
curl 'http://localhost:6333/collections'
```

### Python Client

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collections = client.get_collections()
print(collections)

# Search with vector
results = client.search(
    collection_name="nthakur.dragon-plus-context-encoder",
    query_vector=[0.1, 0.2, ..., 0.8],
    limit=3,
)
```

### Web Dashboard

Open your browser to `http://localhost:6333/dashboard` to access the Qdrant web UI.

## Data Persistence

The container stores data in the `/qdrant_storage` directory, which should be mounted as a volume for persistence:

```bash
# Create a named volume
docker volume create qdrant_data

# Use it with the container
docker run -d \
  --name qdrant-vector-db \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

## Health Monitoring

The container includes a health check that verifies the Qdrant service is operational:

```bash
# Check container health
docker ps
```

The STATUS column will show "(healthy)" when the service is running properly.

## Troubleshooting

### GPU error: "could not select device driver"

If you get an error like this when using `--gpus all`:

```sh
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

This means the NVIDIA Container Toolkit is not properly installed. To fix it:

```bash
# Install the NVIDIA Container Toolkit
# For Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Container fails to start

If the container fails to start, check the logs:

```bash
docker logs qdrant-vector-db
```

### Insufficient disk space

The image is approximately 3.6GB. If you encounter disk space issues:

```bash
# Check available disk space
df -h

# Clean up unused Docker resources
docker system prune -a
```

### Cannot connect to Qdrant API

Verify the container is running and healthy:

```bash
docker ps
```

Check if the ports are correctly mapped:

```bash
docker port qdrant-vector-db
```

### Knowledge store initialization fails

Try running with `TINY_SAMPLE=true` first to test the initialization process with a smaller dataset:

```bash
docker run -d \
  --name qdrant-vector-db-test \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  -e TINY_SAMPLE=true \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Qdrant](https://qdrant.tech/) - Vector Database
- [Facebook AI Research](https://github.com/facebookresearch/atlas) - Atlas Corpus
