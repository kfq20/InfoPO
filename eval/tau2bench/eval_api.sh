#!/bin/bash
# Evaluation script for API model (GPT-4, etc.) on Tau2Bench

set -e

# ========================================
# Configuration
# ========================================

# Model configuration
MODEL_NAME="gemini-3-flash-preview"  # OpenAI model name
EXPERIMENT_NAME="${MODEL_NAME}_tau2bench_pass4"

# Evaluation configuration
DOMAINS="airline telecom retail"  # Domains to evaluate
TASK_SPLIT="test"  # IMPORTANT: Use 'test' split for evaluation
NUM_TRIALS=4  # Number of trials per task (for Pass@k metrics)
MAX_STEPS=70  # Maximum conversation turns
TEMPERATURE=0.7  # Agent temperature

# User simulator configuration
USER_MODEL="gpt-4o-mini-2024-07-18"  # Model for simulating users
USER_BASE_URL=""  # User simulator API endpoint
USER_TEMPERATURE=0.0  # User temperature

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="outputs/tau2bench"
TAU2_DATA_DIR=""  # TODO: Update this path

# ========================================
# Environment Setup
# ========================================

# Set OpenAI API key
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
# Set tau2 data directory
export TAU2_DATA_DIR="${TAU2_DATA_DIR}"

# Disable litellm logging
export LITELLM_LOG="ERROR"

# ========================================
# Pre-flight Checks
# ========================================

echo "========================================="
echo "Tau2Bench Evaluation - API Model"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Domains: ${DOMAINS}"
echo "  Task Split: ${TASK_SPLIT}"
echo "  Num Trials: ${NUM_TRIALS}"
echo "  User Model: ${USER_MODEL}"
echo "========================================="
echo ""

# Check if tau2 is installed
if ! command -v tau2 &> /dev/null; then
    echo "✗ tau2 command not found"
    echo "Please install tau2-bench: pip install -e /path/to/tau2-bench"
    exit 1
fi
echo "✓ tau2 is installed"

# Check tau2 data directory
if [ ! -d "${TAU2_DATA_DIR}" ]; then
    echo "✗ Tau2 data directory not found: ${TAU2_DATA_DIR}"
    echo "Please update TAU2_DATA_DIR in this script"
    exit 1
fi
echo "✓ Tau2 data directory found"

# Check OpenAI API key
if [ -z "${OPENAI_API_KEY}" ] || [ "${OPENAI_API_KEY}" = "your-openai-key" ]; then
    echo "✗ OPENAI_API_KEY not set"
    echo "Please set your OpenAI API key"
    exit 1
fi
echo "✓ OpenAI API key is set"

echo ""
echo "All checks passed! Starting evaluation..."
echo ""

# ========================================
# Run Evaluation
# ========================================

python3 "${SCRIPT_DIR}/run_tau2_eval.py" \
    --evaluate \
    --model-name "${MODEL_NAME}" \
    --backend openai \
    --temperature ${TEMPERATURE} \
    --domains ${DOMAINS} \
    --task-split "${TASK_SPLIT}" \
    --num-trials ${NUM_TRIALS} \
    --max-steps ${MAX_STEPS} \
    --user-model "${USER_MODEL}" \
    --user-base-url "${USER_BASE_URL}" \
    --user-temperature ${USER_TEMPERATURE} \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --tau2-data-dir "${TAU2_DATA_DIR}"

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo ""
echo "To analyze results, run:"
echo "  ./analyze.sh --experiments ${EXPERIMENT_NAME}"
echo "========================================="
