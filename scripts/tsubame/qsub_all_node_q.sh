# Your env vars
REPO_PATH="/gs/fs/tga-okazaki/saito/lighteval_swallow_dev/swallow-evaluation-instruct-private"
HUGGINGFACE_CACHE="/gs/bs/tga-okazaki/saito/HF_HOME"

# Model-dependent vars
MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_MESSAGE="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_CONTEXT_WINDOW=8192

# Other paths
RESULTS_PATH="${REPO_PATH}/results/${MODEL_NAME}"
SCRIPTS_PATH="${REPO_PATH}/scripts/tsubame"


# Japanese
# JMMLU
JMMLU_RESULT_PATH="${RESULTS_PATH}/ja/jmmlu"
mkdir -p "${JMMLU_RESULT_PATH}"
qsub -g tga-okazaki -o "${JMMLU_RESULT_PATH}/logs/" -e "${JMMLU_RESULT_PATH}/logs/" -l node_q=1 -l h_rt=01:00:00 "${SCRIPTS_PATH}/evaluate_ja_jmmlu.sh" "${REPO_PATH}" "${HUGGINGFACE_CACHE}" "${JMMLU_RESULT_PATH}" "${MODEL_NAME}" "${SYSTEM_MESSAGE}" "${MAX_CONTEXT_WINDOW}"
