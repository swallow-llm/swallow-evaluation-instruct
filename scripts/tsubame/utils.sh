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
    export VLLM_CACHE_ROOT=$VLLM_CACHE
    export REPO_PATH=$REPO_PATH

    # Login to HuggingFace
    source "${REPO_PATH}/.common_envs/bin/activate"
    huggingface-cli login --token $HF_TOKEN
    deactivate

    # Special Settings
    ## CUDA_LAUNCH_BLOCKING - to prevent evaluation from stopping at a certain batch. (Default: 0)
    ## (This setting should be done only if necessary because it might slow evaluation.)
    ## (This setting is overwritten with 1 only when evaluating XL-Sum.)
    # export CUDA_LAUNCH_BLOCKING=1 && echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"

    ## UV Options
    UV_OPTIONS="--isolated --project ${REPO_PATH} --locked"

    echo "✅ Common initialization was successfully done."
}


serve_litellm(){
    # Load Args
    MODEL_NAME=$1
    PROVIDER=$2
    REPO_PATH=$3
    GEN_PARAMS=$4
    TASK_NAME=$5
    NUM_GPUS=$6
    GPU_MEMORY_UTILIZATION=$7
    MAX_MODEL_LENGTH=${8:-}

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
            ;;
        *)
            echo "💀 Error: Invalid provider. Must be one of: openai, deepinfra, vllm."
            exit 1
            ;;
    esac
    mkdir -p "$RAW_DIR"

    # Set log level to WARNING
    export LITELLM_LOG_LEVEL=WARNING

    # Start VLLM server
    if [ "$PROVIDER" = "vllm" ]; then
        ## Detect max length and reasoning-tag
        readarray -t results < <(uv run --isolated --project ${REPO_PATH}  --locked --extra auto_detector python - "$MODEL_NAME" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer

model_id = sys.argv[1]

# Detect max length from config
## Get config (trust_remote_code=True is required for most community models)
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

## Search for candidate keys in order
for key in (
    "max_position_embeddings",
    "model_max_length",
    "context_length",
    "seq_length"
):
    val = getattr(cfg, key, None)
    if isinstance(val, int) and val > 0:
        break
else:
    val = 32768

# Reasoning-tag Detection
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
vocab = tokenizer.get_vocab()

think_tags = [
    ## Add think tags here
    "<think>", "</think>",
]
has_think_tag = any(vocab.get(tag) is not None for tag in think_tags)

print(min(32768, val))
print(str(has_think_tag).lower())
PY
)

        ## Set max model length if not specified
        if [[ $MAX_MODEL_LENGTH == "-1" ]]; then
            MAX_MODEL_LENGTH="${results[0]}"
            echo "🤖 (Auto-detected) MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}"
        else
            echo "🖐️ (Manually specified) MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}"
        fi

        ## Set reasoning-tag parameter
        if [ "${results[1]}" = "true" ]; then
            REASONING_TAG_PARAMS="--reasoning-tag"
            echo "🤖 (Auto-detected) reasoning-tag option is set."
        else
            REASONING_TAG_PARAMS=""
            echo "🤖 (Auto-detected) reasoning-tag option is not set."
        fi

        ## Start vllm server in background
        echo "🏗️ Starting vllm server..."
        uv run --isolated --project ${REPO_PATH} --locked --extra vllm \
            vllm serve \
                $MODEL_NAME \
                --tensor-parallel-size $NUM_GPUS \
                --max-model-len $MAX_MODEL_LENGTH \
                --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                --dtype bfloat16 \
                $REASONING_TAG_PARAMS \
                1>&2 &

        ## Wait for server to start
        echo "🔍 Waiting for vllm server to start..."
        TIMEOUT=3600
        start_time=$(date +%s)
        until curl -fs "${BASE_URL%/v1}/health" > /dev/null; do
            sleep 1
            if (( $(date +%s) - start_time >= TIMEOUT )); then
                echo "❌ vLLM server did not become ready within ${TIMEOUT} seconds." >&2
                exit 1
            fi
        done
        end_time=$(date +%s)
        wait_time=$((end_time - start_time))
        echo "✅ vLLM server is ready (took ${wait_time} seconds)"
    fi

    # Create YAML file
    ## api_key is only needed for openai and deepinfra
    MODEL_CONFIG_PATH="$RAW_DIR/model_config_${TASK_NAME}.yaml"
    cat >"$MODEL_CONFIG_PATH" <<EOL
model:
    base_params:
        model_name: $MODEL_NAME_CONFIG
        base_url: $BASE_URL
$GEN_PARAMS
$( [[ $PROVIDER != vllm ]] && printf "    api_key: %s\n" "$API_KEY" )
EOL
    echo "✅ YAML file is created at $MODEL_CONFIG_PATH."

    # Serving is done
    echo "🎉 Serving is done."
}


aggregate_result(){
    MODEL_NAME=$1
    RAW_OUTPUTS_DIR=$2
    AGGREGATED_OUTPUTS_DIR=$3
    REPO_PATH=$4

    uv run --isolated --project ${REPO_PATH} --locked --extra aggregate_results \
        python ${REPO_PATH}/scripts/aggregate_result.py --model $MODEL_NAME \
        --raw-outputs-dir "${RAW_OUTPUTS_DIR}" \
        --aggregated-outputs-dir $AGGREGATED_OUTPUTS_DIR

    echo "✅ Result aggregation was successfully done."
}