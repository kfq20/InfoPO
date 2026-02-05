#!/bin/bash
# Evaluation script for API model on ColBench
# Both agent and user simulator use API models

set -e

# ========================================
# Configuration
# ========================================

# Agent model (API)
AGENT_MODEL="gemini-3-flash-preview"  # or gpt-4o-mini, etc.
AGENT_API_BASE_URL=""  # or https://api.openai.com/v1
OPENAI_API_KEY=""
AGENT_TEMPERATURE=0.7

# User simulator (API) - same API as agent or different
USER_SIMULATOR_MODEL="gpt-4o-mini-2024-07-18"  # Model for user simulator
USER_SIMULATOR_API_BASE_URL="${AGENT_API_BASE_URL}"  # Can use same or different API
USER_SIMULATOR_API_KEY="${OPENAI_API_KEY}"  # Can use same or different API key

# Experiment settings
EXPERIMENT_NAME="${AGENT_MODEL}_colbench"
NUM_TASKS=100  # Number of tasks from test set
BATCH_SIZE=10  # Parallel environments
MAX_STEPS=10   # Max conversation turns
BEST_OF_N=1    # Samples per task (set >1 for Best-of-k)
TASK_TYPE="code"  # code or html
SEED=""  # Random seed for task sampling (empty = use first num_tasks deterministically)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/colbench"

# ========================================
# Environment Setup
# ========================================

export OPENAI_API_KEY=""

# ========================================
# Pre-flight Checks
# ========================================

echo "========================================"
echo "ColBench Evaluation - API Model"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Agent Model: ${AGENT_MODEL}"
echo "  Agent API: ${AGENT_API_BASE_URL}"
echo "  Agent Temperature: ${AGENT_TEMPERATURE}"
echo ""
echo "  User Simulator Model: ${USER_SIMULATOR_MODEL}"
echo "  User Simulator API: ${USER_SIMULATOR_API_BASE_URL}"
echo ""
echo "  Num Tasks: ${NUM_TASKS}"
echo "  Best-of-${BEST_OF_N}"
echo "  Max Steps: ${MAX_STEPS}"
if [ -n "${SEED}" ]; then
    echo "  Random Seed: ${SEED}"
else
    echo "  Random Seed: None (using first ${NUM_TASKS} tasks)"
fi
echo "========================================"
echo ""

# Check API key
if [ -z "${OPENAI_API_KEY}" ] || [ "${OPENAI_API_KEY}" = "your-api-key" ]; then
    echo "✗ OPENAI_API_KEY not set"
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='your-key'"
    exit 1
fi
echo "✓ API key is set"

# Check data
DATA_PATH="${SCRIPT_DIR}/../../data/colbench_code/test.parquet"
if [ ! -f "${DATA_PATH}" ]; then
    echo "✗ Test data not found: ${DATA_PATH}"
    exit 1
fi
echo "✓ Test data found"

echo ""
echo "Starting evaluation..."
echo ""

# ========================================
# Step 1: Generate Trajectories
# ========================================

echo "Step 1: Generating trajectories..."
echo ""

python3 "${SCRIPT_DIR}/run_simulate_api.py" \
    --agent_model="${AGENT_MODEL}" \
    --agent_api_key="${OPENAI_API_KEY}" \
    --agent_base_url="${AGENT_API_BASE_URL}" \
    --agent_temperature=${AGENT_TEMPERATURE} \
    --user_simulator_model="${USER_SIMULATOR_MODEL}" \
    --user_simulator_api_key="${USER_SIMULATOR_API_KEY}" \
    --user_simulator_base_url="${USER_SIMULATOR_API_BASE_URL}" \
    --output_dir="${OUTPUT_DIR}" \
    --experiment_name="${EXPERIMENT_NAME}" \
    --num_tasks=${NUM_TASKS} \
    --batch_size=${BATCH_SIZE} \
    --max_steps=${MAX_STEPS} \
    --best_of_n=${BEST_OF_N} \
    --task_type="${TASK_TYPE}" \
    $([ -n "${SEED}" ] && echo "--seed=${SEED}" || echo "")

# ========================================
# Step 2: Evaluate Code
# ========================================

TRAJECTORY_FILE="${OUTPUT_DIR}/${EXPERIMENT_NAME}/trajectories.jsonl"

echo ""
echo "Step 2: Evaluating code correctness..."
echo ""

python3 "${SCRIPT_DIR}/run_evaluate.py" \
    --saved_path="${TRAJECTORY_FILE}" \
    --k=${BEST_OF_N}

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}/"
echo ""
echo "Files:"
echo "  - trajectories.jsonl: Dialogue histories and generated code"
echo "  - (rewards added to same file)"
echo ""
echo "To analyze results:"
echo "  python -c \"import json; results = [json.loads(l) for l in open('${TRAJECTORY_FILE}')]; print(f'Avg reward: {sum(r[\\\"reward\\\"] for r in results)/len(results):.3f}')\""
echo "========================================"
