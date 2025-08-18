wait_for_vllm_server() {
    local base_url=$1
    local vllm_server_pid=$2
    local timeout=$3

    local start_time=$(date +%s)
    until curl -fs "${base_url%/v1}/health" >/dev/null 2>&1; do
        if ! kill -0 "$vllm_server_pid" 2>/dev/null || (( $(date +%s) - start_time >= timeout )); then
            echo "[error] vLLM server failed to start."
            exit 1
        fi
        sleep 1
    done
}