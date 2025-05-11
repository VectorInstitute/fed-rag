#!/bin/bash
set -e

# Set up signal handling for cleanup
cleanup() {
    echo "Received shutdown signal - cleaning up..."
    # Kill any background processes we may have started
    if [ ! -z "$QDRANT_PID" ]; then
        echo "Stopping temporary Qdrant process..."
        kill $QDRANT_PID 2>/dev/null || true
    fi
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT

echo "Starting Qdrant Atlas Knowledge Store container"

# Make sure we're in the right directory
if [ -d "/qdrant" ]; then
    cd /qdrant
else
    echo "ERROR: /qdrant directory not found!"
    exit 1
fi

# Check if the Qdrant binary exists and is executable
if [ ! -x "./qdrant" ]; then
    echo "ERROR: Qdrant binary not found or not executable!"
    exit 1
fi

# Verify healthcheck.sh exists and is executable
if [ ! -x "/app/healthcheck.sh" ]; then
    echo "WARNING: /app/healthcheck.sh not found or not executable!"
    echo "Making it executable..."
    chmod +x /app/healthcheck.sh
fi

# Default model parameters
MODEL_NAME=${MODEL_NAME:-""}
QUERY_MODEL_NAME=${QUERY_MODEL_NAME:-"nthakur/dragon-plus-query-encoder"}
CONTEXT_MODEL_NAME=${CONTEXT_MODEL_NAME:-"nthakur/dragon-plus-context-encoder"}
BATCH_SIZE=${BATCH_SIZE:-5000}
CLEAR_FIRST=${CLEAR_FIRST:-True}
FILENAME=${FILENAME:-"text-list-100-sec.jsonl"}
CORPUS=${CORPUS:-"enwiki-dec2021"}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-"false"}
TINY_SAMPLE=${TINY_SAMPLE:-"false"}
SAMPLE_SIZE=${SAMPLE_SIZE:-"full"}  # Options: "tiny", "small", "full"

# Handle backward compatibility
if [ "$TINY_SAMPLE" = "true" ]; then
    SAMPLE_SIZE="tiny"
fi

# Identify Qdrant storage path
QDRANT_STORAGE_PATH="/qdrant_storage"
QDRANT_COLLECTIONS_PATH="$QDRANT_STORAGE_PATH/collections"

# Create storage directory if it doesn't exist
mkdir -p $QDRANT_STORAGE_PATH

# Define a function to check if Qdrant storage is already initialized
check_qdrant_initialized() {
    if [ -d "$QDRANT_COLLECTIONS_PATH" ] && [ "$(ls -A $QDRANT_COLLECTIONS_PATH 2>/dev/null)" ]; then
        return 0  # true - initialized
    fi
    return 1  # false - not initialized
}

# Create a very small sample file for testing
create_tiny_sample() {
    echo "Creating tiny sample file for testing..."
    mkdir -p /app/data/atlas/$CORPUS

    # Create a small JSON Lines file with 10 sample entries
    cat > /app/data/atlas/$CORPUS/tiny-sample.jsonl << EOF
{"id": "140", "title": "History of marine biology", "section": "James Cook", "text": " James Cook is well known for his voyages of exploration for the British Navy in which he mapped out a significant amount of the world's uncharted waters. Cook's explorations took him around the world twice and led to countless descriptions of previously unknown plants and animals. Cook's explorations influenced many others and led to a number of scientists examining marine life more closely. Among those influenced was Charles Darwin who went on to make many contributions of his own. "}
{"id": "141", "title": "History of marine biology", "section": "Charles Darwin", "text": " Charles Darwin, best known for his theory of evolution, made many significant contributions to the early study of marine biology. He spent much of his time from 1831 to 1836 on the voyage of HMS Beagle collecting and studying specimens from a variety of marine organisms. It was also on this expedition where Darwin began to study coral reefs and their formation. He came up with the theory that the overall growth of corals is a balance between the growth of corals upward and the sinking of the sea floor. He then came up with the idea that wherever coral atolls would be found, the central island where the coral had started to grow would be gradually subsiding"}
{"id": "142", "title": "History of marine biology", "section": "Charles Wyville Thomson", "text": " Another influential expedition was the voyage of HMS Challenger from 1872 to 1876, organized and later led by Charles Wyville Thomson. It was the first expedition purely devoted to marine science. The expedition collected and analyzed thousands of marine specimens, laying the foundation for present knowledge about life near the deep-sea floor. The findings from the expedition were a summary of the known natural, physical and chemical ocean science to that time."}
{"id": "143", "title": "History of marine biology", "section": "Later exploration", "text": " This era of marine exploration came to a close with the first and second round-the-world voyages of the Danish Galathea expeditions and Atlantic voyages by the USS Albatross, the first research vessel purpose built for marine research. These voyages further cleared the way for modern marine biology by building a base of knowledge about marine biology. This was followed by the progressive development of more advanced technologies which began to allow more extensive explorations of ocean depths that were once thought too deep to sustain life."}
{"id": "144", "title": "History of marine biology", "section": "Marine biology labs", "text": " In the 1960s and 1970s, ecological research into the life of the ocean was undertaken at institutions set up specifically to study marine biology. Notable was the Woods Hole Oceanographic Institution in America, which established a model for other marine laboratories subsequently set up around the world. Their findings of unexpectedly high species diversity in places thought to be inhabitable stimulated much theorizing by population ecologists on how high diversification could be maintained in such a food-poor and seemingly hostile environment. "}
{"id": "145", "title": "History of marine biology", "section": "Exploration technology", "text": " In the past, the study of marine biology has been limited by a lack of technology as researchers could only go so deep to examine life in the ocean. Before the mid-twentieth century, the deep-sea bottom could not be seen unless one dredged a piece of it and brought it to the surface. This has changed dramatically due to the development of new technologies in both the laboratory and the open sea. These new technological developments have allowed scientists to explore parts of the ocean they didn't even know existed. The development of scuba gear allowed researchers to visually explore the oceans as it contains a self-contained underwater breathing apparatus allowing a person to breathe while being submerged 100 to 200 feet "}
{"id": "146", "title": "History of marine biology", "section": "Exploration technology", "text": " the ocean. Submersibles were built like small submarines with the purpose of taking marine scientists to deeper depths of the ocean while protecting them from increasing atmospheric pressures that cause complications deep under water. The first models could hold several individuals and allowed limited visibility but enabled marine biologists to see and photograph the deeper portions of the oceans. Remotely operated underwater vehicles are now used with and without submersibles to see the deepest areas of the ocean that would be too dangerous for humans. ROVs are fully equipped with cameras and sampling equipment which allows researchers to see and control everything the vehicle does. ROVs have become the dominant type of technology used to view the deepest parts of the ocean."}
{"id": "147", "title": "History of marine biology", "section": "Romanticization", "text": " In the late 20th century and into the 21st, marine biology was \"glorified and romanticized through films and television shows,\" leading to an influx in interested students who required a damping on their enthusiasm with the day-to-day realities of the field."}
{"id": "148", "title": "Wynthryth", "section": "", "text": " Wynthryth of March was an early medieval saint of Anglo Saxon England. He is known to history from the Secgan Hagiography and The Confraternity Book of  St Gallen. Very little is known of his life or career. However, he was associated with the town of March, Cambridgeshire, and he may have been a relative of King Ethelstan."}
{"id": "149", "title": "James M. Safford", "section": "", "text": " James Merrill Safford (1822–1907) was an American geologist, chemist and university professor."}
{"id": "150", "title": "James M. Safford", "section": "Early life", "text": " James M. Safford was born in Putnam, Ohio on August 13, 1822. He received an M.D. and a PhD. He was trained as a chemist at Yale University. He married Catherine K. Owens in 1859, and they had two children."}
{"id": "151", "title": "James M. Safford", "section": "Career", "text": " Safford taught at Cumberland University in Lebanon, Tennessee from 1848 to 1873. He served as a Professor of Mineralogy, Botany, and Economical Geology at Vanderbilt University in Nashville, Tennessee from 1875 to 1900. He was a Presbyterian, and often started his lessons with a prayer. He served on the Tennessee Board of Health. Additionally, he acted as a chemist for the Tennessee Bureau of Agriculture in the 1870s and 1880s. He published fifty-four books, reports, and maps."}
{"id": "152", "title": "James M. Safford", "section": "Death", "text": " He died in Dallas on July 2, 1907."}
EOF

    # Set filename to use the tiny sample
    FILENAME="tiny-sample.jsonl"
    echo "Using tiny sample file: $FILENAME"

    echo "Verifying sample file creation..."
    if [ -f "/app/data/atlas/$CORPUS/tiny-sample.jsonl" ]; then
        echo "✅ Sample file successfully created at: /app/data/atlas/$CORPUS/tiny-sample.jsonl"
        echo "File details:"
        ls -la /app/data/atlas/$CORPUS/tiny-sample.jsonl
        echo "File content (first 3 lines):"
        head -n 3 /app/data/atlas/$CORPUS/tiny-sample.jsonl
        echo "Directory listing:"
        ls -la /app/data/atlas/$CORPUS/
    else
        echo "❌ ERROR: Sample file was NOT created at: /app/data/atlas/$CORPUS/tiny-sample.jsonl"
    fi
}

# Download a small sample from Dropbox (approximately 100MB)
download_small_sample() {
    echo "Downloading small sample file from Dropbox..."
    mkdir -p /app/data/atlas/$CORPUS

    # Install curl if not available
    if ! command -v curl &> /dev/null; then
        echo "Installing curl..."
        apt-get update && apt-get install -y curl
    fi

    # Download small sample file (adjust URL as needed)
    SMALL_SAMPLE_URL="https://www.dropbox.com/scl/fi/v030orqudy6672zs5d3ql/sample_1m-text-list-100-sec.jsonl?rlkey=2zbif6m7zbqdw0rwau0sn5dcu&st=46q6kb4o&dl=0"
    SMALL_SAMPLE_FILE="/app/data/atlas/$CORPUS/sample_1m-text-list-100-sec.jsonl"

    echo "Downloading from: $SMALL_SAMPLE_URL"
    curl -L -o "$SMALL_SAMPLE_FILE" "$SMALL_SAMPLE_URL"

    # Set filename to use the small sample
    FILENAME="sample_1m-text-list-100-sec.jsonl"
    echo "Using small sample file: $FILENAME"

    echo "Verifying sample file download..."
    if [ -f "$SMALL_SAMPLE_FILE" ]; then
        echo "✅ Small sample file successfully downloaded to: $SMALL_SAMPLE_FILE"
        echo "File details:"
        ls -la "$SMALL_SAMPLE_FILE"
        echo "File content (first 3 lines):"
        head -n 3 "$SMALL_SAMPLE_FILE"
    else
        echo "❌ ERROR: Small sample file was NOT downloaded successfully"
        echo "Falling back to tiny sample..."
        create_tiny_sample
    fi
}

# Initialization function
initialize_database() {
    # Check if Qdrant is already initialized
    if check_qdrant_initialized; then
        echo "Database already initialized, skipping setup."
        rm -f "/data/initialization_in_progress"
        return
    fi

    # Mark initialization as in progress
    mkdir -p /data
    touch "/data/initialization_in_progress"

    # Handle sample size selection
    if [ "$SAMPLE_SIZE" = "tiny" ]; then
        echo "Using tiny sample mode..."
        create_tiny_sample
    elif [ "$SAMPLE_SIZE" = "small" ]; then
        echo "Using small sample mode..."
        download_small_sample
    elif [ "$SKIP_DOWNLOAD" != "true" ]; then
        echo "Using full corpus mode..."
        # Clone Atlas repository if not already done
        if [ ! -d "/app/atlas" ]; then
            echo "Cloning Atlas repository..."
            cd /app
            git clone --depth 1 https://github.com/facebookresearch/atlas.git /app/atlas

            # Install only wget which is the only requirement for download_corpus.py
            pip install wget

            # Download the corpus
            echo "Downloading full Atlas corpus (this may take a significant amount of time)..."
            cd /app/atlas
            python preprocessing/download_corpus.py --corpus corpora/wiki/$CORPUS --output_directory /data/atlas

            # Create smaller sample file if needed
            if [ "$FILENAME" != "text-list-100-sec.jsonl" ] && [ -f "/data/atlas/$CORPUS/text-list-100-sec.jsonl" ]; then
                echo "Creating sample file from corpus..."
                head -n 10000 /data/atlas/$CORPUS/text-list-100-sec.jsonl > /data/atlas/$CORPUS/$FILENAME
            fi
        fi
    else
        echo "Skipping download as SKIP_DOWNLOAD=true, but no data found."
        echo "You need to mount a volume with data at /data/atlas/$CORPUS/$FILENAME"
        echo "or set SAMPLE_SIZE=tiny to use a built-in tiny sample file."
        mkdir -p /data/atlas/$CORPUS
        if [ ! -f "/data/atlas/$CORPUS/$FILENAME" ]; then
            echo "WARNING: No data file found. Creating tiny sample as fallback."
            create_tiny_sample
        fi
    fi

    # Start Qdrant in the background
    echo "Starting Qdrant service for initialization..."
    cd /qdrant
    ./entrypoint.sh &
    QDRANT_PID=$!

    # Wait for Qdrant to be ready
    echo "Waiting for Qdrant service to be ready..."
    ATTEMPTS=0
    MAX_ATTEMPTS=30
    until /app/healthcheck.sh; do
        ATTEMPTS=$((ATTEMPTS+1))
        if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
            echo "ERROR: Qdrant failed to start after $MAX_ATTEMPTS attempts"
            kill $QDRANT_PID 2>/dev/null || true
            exit 1
        fi
        echo "Waiting for Qdrant to start... (Attempt $ATTEMPTS/$MAX_ATTEMPTS)"
        sleep 5
    done

    echo "Qdrant is ready! Building knowledge store..."

    # Build command with core parameters
    CMD="cd /app && uv run python -m ra_dit_ks.main --clear_first $CLEAR_FIRST --batch_size $BATCH_SIZE --filename \"$FILENAME\""

    # Handle model parameters with proper logic
    if [ ! -z "$MODEL_NAME" ]; then
        # If MODEL_NAME is specified, use it
        CMD="$CMD --model_name \"$MODEL_NAME\""
        echo "Using MODEL_NAME: $MODEL_NAME"
    else
        # Otherwise, use the encoder models if provided
        if [ ! -z "$QUERY_MODEL_NAME" ] || [ ! -z "$CONTEXT_MODEL_NAME" ]; then
            if [ ! -z "$QUERY_MODEL_NAME" ]; then
                CMD="$CMD --query_model_name \"$QUERY_MODEL_NAME\""
                echo "Using QUERY_MODEL_NAME: $QUERY_MODEL_NAME"
            fi

            if [ ! -z "$CONTEXT_MODEL_NAME" ]; then
                CMD="$CMD --context_model_name \"$CONTEXT_MODEL_NAME\""
                echo "Using CONTEXT_MODEL_NAME: $CONTEXT_MODEL_NAME"
            fi
        fi
    fi

    # Build the QdrantKnowledgeStore with specified parameters
    echo "Building QdrantKnowledgeStore with command:"
    echo "$CMD"

    # Execute the command
    eval $CMD

    # Shutdown the temporary Qdrant process
    echo "Knowledge store build complete, stopping temporary Qdrant service..."
    kill $QDRANT_PID
    wait $QDRANT_PID || true

    # Remove initialization marker
    rm -f "/data/initialization_in_progress"

    echo "Initialization completed successfully!"
}

# Run initialization function
echo "Running database initialization check..."
initialize_database

# Add a delay to ensure resources are released
echo "Waiting for resources to be released..."
sleep 5

# Start Qdrant in the foreground
echo "Starting Qdrant service with initialized data..."
cd /qdrant  # Ensure we're in the right directory
exec ./entrypoint.sh
