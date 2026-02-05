#!/bin/bash
# Evaluation script for VLLM model on ColBench
# Agent uses VLLM server, user simulator uses API

set -e

# ========================================
# Configuration
# ========================================

# Agent model (VLLM server)
AGENT_VLLM_HOST="localhost:8500"  # Host:port of agent VLLM server
AGENT_MODEL="inforl_colbench"  # Model name on VLLM server (custom name)
AGENT_TEMPERATURE=0.0

# User simulator (API) - same API as eval_api.sh
USER_SIMULATOR_MODEL="gpt-4o-mini-2024-07-18"  # Model for user simulator
USER_SIMULATOR_API_BASE_URL=""  # or https://api.openai.com/v1
OPENAI_API_KEY=""

# Experiment settings
EXPERIMENT_NAME="${AGENT_MODEL}_colbench"
NUM_TASKS=1000  # Number of tasks from test set
BATCH_SIZE=32  # Parallel environments
MAX_STEPS=10   # Max conversation turns
BEST_OF_N=1    # Samples per task (set >1 for Best-of-k)
TASK_TYPE="code"  # code or html

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/colbench"

# ========================================
# Environment Setup
# ========================================

export OPENAI_API_KEY="${OPENAI_API_KEY}"

# ========================================
# Pre-flight Checks
# ========================================

echo "========================================"
echo "ColBench Evaluation - VLLM Agent + API User Simulator"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Agent Model: ${AGENT_MODEL}"
echo "  Agent VLLM Server: ${AGENT_VLLM_HOST}"
echo "  Agent Temperature: ${AGENT_TEMPERATURE}"
echo ""
echo "  User Simulator Model: ${USER_SIMULATOR_MODEL}"
echo "  User Simulator API: ${USER_SIMULATOR_API_BASE_URL}"
echo ""
echo "  Num Tasks: ${NUM_TASKS}"
echo "  Best-of-${BEST_OF_N}"
echo "  Max Steps: ${MAX_STEPS}"
echo "========================================"
echo ""

# Check agent VLLM server
AGENT_VLLM_URL="http://localhost:8500/v1"
if ! curl -s "${AGENT_VLLM_URL}" > /dev/null 2>&1; then
    echo "✗ Agent VLLM server not accessible at ${AGENT_VLLM_HOST}"
    echo ""
    echo "Please start the agent VLLM server first:"
    echo "  vllm serve <your-model-path> \\"
    echo "    --port ${AGENT_VLLM_HOST##*:} \\"
    echo "    --tensor-parallel-size 8"
    exit 1
fi
echo "✓ Agent VLLM server is running"

# Check API key
if [ -z "${OPENAI_API_KEY}" ] || [ "${OPENAI_API_KEY}" = "your-api-key" ]; then
    echo "✗ OPENAI_API_KEY not set"
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='your-key'"
    exit 1
fi
echo "✓ API key is set"

# Check data
DATA_PATH="${SCRIPT_DIR}/../../data/colbench_code/train.parquet"
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
    --agent_vllm_host="${AGENT_VLLM_HOST}" \
    --agent_temperature=${AGENT_TEMPERATURE} \
    --user_simulator_model="${USER_SIMULATOR_MODEL}" \
    --user_simulator_api_key="${OPENAI_API_KEY}" \
    --user_simulator_base_url="${USER_SIMULATOR_API_BASE_URL}" \
    --output_dir="${OUTPUT_DIR}" \
    --experiment_name="${EXPERIMENT_NAME}" \
    --num_tasks=${NUM_TASKS} \
    --batch_size=${BATCH_SIZE} \
    --max_steps=${MAX_STEPS} \
    --best_of_n=${BEST_OF_N} \
    --task_type="${TASK_TYPE}"

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
