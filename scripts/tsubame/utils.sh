#!/bin/bash

init_common(){
    # Load Modules
    . /etc/profile.d/modules.sh

    # Load Args
    MODEL_NAME=$1
    NODE_KIND=$2
    REPO_PATH=$3
    
    # Load and Set Vars
    source "${REPO_PATH}/.env"
    export PATH="$HOME/.local/bin:$PATH" # for uv, add local bin to PATH
    export TMPDIR=/local/${JOB_ID}
    export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
    export HF_HOME=$HUGGINGFACE_CACHE
    export UV_CACHE_DIR=$UV_CACHE
    export REPO_PATH=$REPO_PATH

    # Login to HuggingFace
    source "${REPO_PATH}/.common_envs/bin/activate"
    huggingface-cli login --token $HF_TOKEN
    deactivate

    # Special Settings
    ## CUDA_LAUNCH_BLOCKING - to prevent evaluation from stopping at a certain batch. (Default: 0)
    ## (This setting should be done only if necessary because it might slow evaluation.)
    ## (This setting is overwritten with 1 only when evaluating XL-Sum.)
    # export CUDA_LAUNCH_BLOCKING=1
    # echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"

    ## UV Options
    UV_OPTIONS="--isolated --locked --index-strategy unsafe-best-match"

    echo "✅ Common initialization was successfully done."
}


serve_litellm(){
    # Load Args
    MODEL_NAME=$1
    PROVIDER=$2
    REPO_PATH=$3

    # Setup based on provider
    RAW_OUTPUT_DIR="${REPO_PATH}/lighteval/outputs"
    case $PROVIDER in
        "openai")
            BASE_URL="https://api.openai.com/v1"
            MODEL_NAME_CONFIG=$MODEL_NAME
            RAW_DIR="$RAW_OUTPUT_DIR/results/$MODEL_NAME"
            ;;
        "deepinfra")
            BASE_URL="https://api.deepinfra.com/v1/openai"
            MODEL_NAME_CONFIG="deepinfra/$MODEL_NAME"
            RAW_DIR="$RAW_OUTPUT_DIR/results/deepinfra/$MODEL_NAME"
            ;;
        "vllm")
            BASE_URL="http://localhost:8000/v1"
            MODEL_NAME_CONFIG="hosted_vllm/$MODEL_NAME"
            RAW_DIR="$RAW_OUTPUT_DIR/results/hosted_vllm/$MODEL_NAME"
            export CUDA_VISIBLE_DEVICES=$GPUS
            ;;
        *)
            echo "💀 Error: Invalid provider. Must be one of: openai, deepinfra, vllm"
            exit 1
            ;;
    esac
    mkdir -p "$RAW_DIR"

    # Set log level to WARNING
    export LITELLM_LOG_LEVEL=WARNING

    # Start VLLM server
    if [ "$PROVIDER" = "vllm" ]; then
        # Cleanup function
        cleanup() {
            if [ ! -z "$VLLM_PID" ]; then
                echo "🧹 Stopping existing vllm server..."
                kill $VLLM_PID
            fi
        }

        # Execute cleanup on script exit
        trap cleanup EXIT

        # Start vllm server in background
        echo "🏗️ Starting vllm server..."
        uv run vllm serve $MODEL_NAME --tensor-parallel-size $NUM_GPUS > "$RAW_DIR/vllm_server.log" 2>&1 &
        VLLM_PID=$!

        # Wait for server to start
        echo "🔍 Waiting for vllm server to start..."
        while ! curl -s $BASE_URL > /dev/null; do
            sleep 1
        done
        echo "✅ vllm server is ready"
    fi


    # Create YAML file
    ## api_key is only needed for openai and deepinfra
    cat >"$RAW_DIR/model_config.yaml" <<EOF
model:
  base_params:
    model_name: $MODEL_NAME_CONFIG
    base_url: $BASE_URL
$( [[ $PROVIDER != vllm ]] && printf "    api_key: %s\n" "$API_KEY" )
EOF
    echo "✅ YAML file is created."

    # Serving is done
    echo "🎉 Serving is done."
}


aggregate_result(){
    MODEL_NAME=$1
    RAW_OUTPUTS_DIR=$2
    AGGREGATED_OUTPUTS_DIR=$3

    uv run --isolated --extra aggregate_results --index-strategy unsafe-best-match \
        python scripts/aggregate_result.py --model $MODEL_NAME \
        --raw-outputs-dir "${RAW_OUTPUTS_DIR}" \
        --aggregated-outputs-dir $AGGREGATED_OUTPUTS_DIR

    echo "✅ Result aggregation was successfully done."
}