#!/bin/bash

init_common(){
    # Global variables which will be defined and become available after this function is over:
    # - NUM_GPUS
    # - GPU_MEMORY_UTILIZATION
    # - UV_OPTIONS

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
    export HF_TOKEN=$HF_TOKEN
    export OPENAI_API_KEY=$OPENAI_API_KEY
    export DEEPINFRA_API_KEY=$DEEPINFRA_API_KEY
    export UV_CACHE_DIR=$UV_CACHE
    export VLLM_CACHE_ROOT=$VLLM_CACHE
    export REPO_PATH=$REPO_PATH

    # GPU Settings
    ## Set NUM_GPUS based on NODE_KIND
    case $NODE_KIND in
        "node_q") NUM_GPUS=1 ;;
        "node_f") NUM_GPUS=4 ;;
        *"cpu"*) NUM_GPUS=0 ;;
        *) echo "❌ Unknown NODE_KIND: $NODE_KIND"
            exit 1
    esac
    ## Set GPU_MEMORY_UTILIZATION
    GPU_MEMORY_UTILIZATION=0.9

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


get_generation_params(){
    # Global variables which will be defined and become available after this function is over:
    # - CUSTOM_SETTINGS_PATH
    # - CUSTOM_SETTINGS_VERSION
    # - CUSTOM_SETTINGS_SUBDIR
    # - CUSTOM_SETTINGS_NAME
    # - MAX_MODEL_LENGTH
    # - REASONING_PARSER
    # - SYSTEM_MESSAGE (Optional: will be included in OPTIONAL_ARGS_FOR_LIGHTEVAL)
    # - GEN_PARAMS
    # - OPTIONAL_ARGS_FOR_LIGHTEVAL

    # Load Args
    CUSTOM_SETTINGS=$1
    TASK_NAME=$2
    REPO_PATH=$3
    MODEL_NAME=$4

    # Load custom settings from model_conf.yaml
    local GENERATION_SETTINGS_DIR="${REPO_PATH}/scripts/generation_settings"
    local TASK_SETTINGS_PATH="${GENERATION_SETTINGS_DIR}/task_settings.csv"
    local CUSTOM_MODEL_SETTINGS_DIR="${GENERATION_SETTINGS_DIR}/custom_model_settings"
    readarray -t results < <(uv run --isolated --project ${REPO_PATH} --locked --extra auto_detector python "${GENERATION_SETTINGS_DIR}/setting_manager.py" --model_id "$MODEL_NAME" --task_id "$TASK_NAME" --custom_settings "$CUSTOM_SETTINGS" --task_settings_path "$TASK_SETTINGS_PATH" --custom_model_settings_dir "$CUSTOM_MODEL_SETTINGS_DIR")
    ## The following scripts are run only when the desired custom settings are found; or no settings are specified.

    CUSTOM_SETTINGS_PATH=${results[0]}
    CUSTOM_SETTINGS_VERSION=${results[1]}
    if [[ $CUSTOM_SETTINGS != "" ]]; then
        CUSTOM_SETTINGS_SUBDIR="/${CUSTOM_SETTINGS}"
        CUSTOM_SETTINGS_NAME="${CUSTOM_SETTINGS}"
        echo "🔍 Custom settings '${CUSTOM_SETTINGS}' is found. Use the settings."
    else
        CUSTOM_SETTINGS_SUBDIR=""
        CUSTOM_SETTINGS_NAME=""
        echo "➖ No custom settings is specified. Use default settings."
    fi

    # Accept SHELL_OUTPUT_PARAMETERS
    MAX_MODEL_LENGTH=${results[2]}
    REASONING_PARSER=${results[3]}
    export VLLM_USE_V1=${results[4]}
    SYSTEM_MESSAGE=${results[5]}

    # Accept CONFIG_YAML_PARAMETERS
    rest=( "${results[@]:6}" )
    GEN_PARAMS=$(printf '%s\n' "${rest[@]}")

    # Prepare optional arguments for lighteval
    OPTIONAL_ARGS_FOR_LIGHTEVAL=""
    if [[ $SYSTEM_MESSAGE != "" ]]; then
        OPTIONAL_ARGS_FOR_LIGHTEVAL+="--system-prompt ${SYSTEM_MESSAGE}"
    fi
}


serve_litellm(){
    # Global variables which will be defined and become available after this function is over:
    # - RAW_OUTPUT_DIR
    # - RAW_RESULT_DIR
    # - MODEL_NAME_CONFIG
    # - MAX_MODEL_LENGTH
    # - MODEL_CONFIG_PATH
    # - VLLM_SERVER_PID
    
    # Load Args
    MODEL_NAME=$1
    PROVIDER=$2
    REPO_PATH=$3
    CUSTOM_SETTINGS_SUBDIR=$4
    GEN_PARAMS=$5
    TASK_NAME=$6
    NODE_KIND=$7
    NUM_GPUS=$8
    GPU_MEMORY_UTILIZATION=$9
    MAX_MODEL_LENGTH=${10}
    REASONING_PARSER=${11}

    # Setup based on provider
    RAW_OUTPUT_DIR="${REPO_PATH}/lighteval/outputs"
    case $PROVIDER in
        "openai")
            local BASE_URL="https://api.openai.com/v1"
            MODEL_NAME_CONFIG="$MODEL_NAME"
            RAW_RESULT_DIR="$RAW_OUTPUT_DIR/results/$MODEL_NAME$CUSTOM_SETTINGS_SUBDIR"
            ;;
        "deepinfra")
            local BASE_URL="https://api.deepinfra.com/v1/openai"
            MODEL_NAME_CONFIG="deepinfra/$MODEL_NAME"
            RAW_RESULT_DIR="$RAW_OUTPUT_DIR/results/deepinfra/$MODEL_NAME$CUSTOM_SETTINGS_SUBDIR"
            ;;
        "vllm")
            local BASE_URL="http://localhost:8000/v1"
            MODEL_NAME_CONFIG="hosted_vllm/$MODEL_NAME"
            RAW_RESULT_DIR="$RAW_OUTPUT_DIR/results/hosted_vllm/$MODEL_NAME$CUSTOM_SETTINGS_SUBDIR"
            ;;
        *)
            echo "💀 Error: Invalid provider. Must be one of: openai, deepinfra, vllm."
            exit 1
            ;;
    esac
    mkdir -p "$RAW_RESULT_DIR"

    # Set log level to WARNING
    export LITELLM_LOG_LEVEL=WARNING

    # Start VLLM server
    if [ "$PROVIDER" = "vllm" ]; then
        if [[ $NODE_KIND == *"cpu"* ]]; then
            echo "❌ VLLM server cannot be started on CPU nodes. Use GPU nodes (node_q or node_f)."
            exit 1
        fi

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
        REASONING_PARSER_PARAM=""
        if [ "${results[1]}" = "true" ]; then
            echo "🤖 (Auto-detected) The specified model has reasoning-tag in its vocabulary."
            if [ "${REASONING_PARSER}" == "ignore" ]; then
                echo "✅ The specified model may support reasoning-tag, but it is ignored. No reasoning-parser will be used."
            elif [ "${REASONING_PARSER}" == ""]; then
                echo "💀 Error: The specified model may support reasoning-tag, but no reasoning-parser is specified. Please specify a reasoning-parser(, or set REASONING_PARSER=ignore if you do not want to use reasoning-tag)."
                exit 1
            else
                echo "✅ Reasoning-parser: ${REASONING_PARSER} is used."
                REASONING_PARSER_PARAM="--reasoning-parser ${REASONING_PARSER}"
            fi
        else
            echo "🤖 (Auto-detected) The specified model does not have reasoning-tag in its vocabulary."
            if [ "${REASONING_PARSER}" == "ignore" ]; then
                echo "✅ No reasoning-parser will be used. (You do not have to specify reasoning_parser=ignore in this case. Please unset reasoning_parser in the YAML file to make it clear.)"
            elif [ "${REASONING_PARSER}" != "" ]; then
                echo "⚠️ Warning: The specified model may not support reasoning-tag, but a reasoning-parser, ${REASONING_PARSER}, is specified and will be used. This may cause an error or unexpected behavior."
            else
                echo "✅ No reasoning-parser will be used."
            fi
        fi

        ## Start vllm server in background
        echo "🏗️ Starting vllm server..."
        uv run --isolated --project ${REPO_PATH} --locked --extra vllm \
            vllm serve \
                $MODEL_NAME \
                --hf-token $HF_TOKEN \
                --tensor-parallel-size $NUM_GPUS \
                --max-model-len $MAX_MODEL_LENGTH \
                --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                --dtype bfloat16 \
                $REASONING_PARSER_PARAM \
                1>&2 &
        VLLM_SERVER_PID=$!

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
    
    else
        if [[ $NODE_KIND == "node_q" || $NODE_KIND == "node_f" ]]; then
            echo "❌ You specified ${NODE_KIND} but OpenAI and DeepInfra do not use GPUs. Use CPU nodes instead."
            exit 1
        fi

        case $PROVIDER in
            "openai") API_KEY=$OPENAI_API_KEY ;;
            "deepinfra") API_KEY=$DEEPINFRA_API_KEY ;;
            *) echo "❌ Invalid provider. Must be one of: openai, deepinfra."
                exit 1
                ;;
        esac
    fi

    # Create YAML file
    ## api_key is only needed for openai and deepinfra
    MODEL_CONFIG_PATH="$RAW_RESULT_DIR/model_config_${TASK_NAME}.yaml"
    cat >"$MODEL_CONFIG_PATH" <<EOL
model:
    base_params:
        model_name: $MODEL_NAME_CONFIG
        base_url: $BASE_URL
$( [[ $PROVIDER != vllm ]] && printf "        api_key: %s\n" "$API_KEY" )
$GEN_PARAMS
EOL
    echo "✅ YAML file is created at $MODEL_CONFIG_PATH."

    # Serving is done
    echo "🎉 Serving is done."
}


stop_vllm_server(){
    # Load Args
    local vllm_pid=$1

    # Stop vllm server
    kill $vllm_pid 2>/dev/null

    echo "🔍 Waiting for VLLM server to stop..."
    KILL_WAIT_TIME=10
    for ((i=0; i<KILL_WAIT_TIME; i++)); do
        kill -0 "$vllm_pid" 2>/dev/null || break
        sleep 1
    done

    if kill -0 "$vllm_pid" 2>/dev/null; then
        echo "❌ VLLM server is still running. Force killing it..."
        kill -9 "$vllm_pid"
    fi
    echo "✅ VLLM server is stopped."
}


aggregate_result(){
    MODEL_NAME=$1
    RAW_OUTPUTS_DIR=$2
    AGGREGATED_OUTPUTS_DIR=$3
    REPO_PATH=$4
    CUSTOM_SETTINGS_PATH=$5
    CUSTOM_SETTINGS_NAME=$6
    CUSTOM_SETTINGS_VERSION=$7

    uv run --isolated --project ${REPO_PATH} --locked --extra aggregate_results \
        python ${REPO_PATH}/scripts/aggregate_results.py \
        --model_name "${MODEL_NAME}" \
        --raw_results_dir "${RAW_OUTPUTS_DIR}" \
        --aggregated_outputs_dir "${AGGREGATED_OUTPUTS_DIR}" \
        --used_custom_settings_path "${CUSTOM_SETTINGS_PATH}" \
        --used_custom_settings_name "${CUSTOM_SETTINGS_NAME}" \
        --used_custom_settings_version "${CUSTOM_SETTINGS_VERSION}"

    echo "✅ Result aggregation was successfully done."
}