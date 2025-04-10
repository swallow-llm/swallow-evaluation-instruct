# Model-dependent settings
MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_MESSAGE="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_CONTEXT_WINDOW=8192

# Other settings
source "$(dirname "$0")/../user_config.sh"
RESULTS_PATH="${REPO_PATH}/results/${MODEL_NAME}"; SCRIPTS_PATH="${REPO_PATH}/scripts/tsubame"
QSUB_CMD="qsub -g tga-okazaki -l node_q=1"

###########

# Japanese
# JMMLU
JMMLU_RESULT_PATH="${RESULTS_PATH}/ja/jmmlu"
mkdir -p "${JMMLU_RESULT_PATH}"
$QSUB_CMD -l h_rt=01:00:00 -o "${JMMLU_RESULT_PATH}/logs/" -e "${JMMLU_RESULT_PATH}/logs/" "${SCRIPTS_PATH}/evaluate_ja_jmmlu.sh" "${REPO_PATH}" "${HUGGINGFACE_CACHE}" "${JMMLU_RESULT_PATH}" "${MODEL_NAME}" "${SYSTEM_MESSAGE}" "${MAX_CONTEXT_WINDOW}"
