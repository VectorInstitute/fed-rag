#!/bin/bash

# Check if Qdrant API is responding
if curl -s -f -X GET "http://localhost:6333/collections" -H "Content-Type: application/json" > /dev/null; then
    # Qdrant API is responding successfully
    exit 0
else
    # Qdrant API is not responding
    exit 1
fi
