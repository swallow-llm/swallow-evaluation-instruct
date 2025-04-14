# Model-dependent settings
MODEL_NAME="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_MESSAGE="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_CONTEXT_WINDOW=8192

# Other settings
source "$(dirname "$0")/../user_config.sh"
USER_CONFIG_PATH="${REPO_PATH}/scripts/user_config.sh"
SCRIPTS_DIR="${REPO_PATH}/scripts/tsubame"
RAW_OUTPUTS_DIR="${REPO_PATH}/lighteval/outputs"
AGGREGATED_OUTPUTS_DIR="${REPO_PATH}/results/${MODEL_NAME}"
QSUB_CMD="qsub -g tga-okazaki -l node_q=1"

###########

# Japanese
# JMMLU
$QSUB_CMD -l h_rt=00:20:00 -o "${AGGREGATED_OUTPUTS_DIR}/logs/" -e "${AGGREGATED_OUTPUTS_DIR}/logs/" "${SCRIPTS_DIR}/evaluate_ja_jmmlu.sh" "${USER_CONFIG_PATH}" "${RAW_OUTPUTS_DIR}" "${AGGREGATED_OUTPUTS_DIR}" "${MODEL_NAME}" "${SYSTEM_MESSAGE}" "${MAX_CONTEXT_WINDOW}"
