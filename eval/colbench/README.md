# ColBench Evaluation

Simple evaluation framework for ColBench (Collaborative Agent Benchmark) supporting both VLLM and API models.

## Overview

ColBench evaluates code generation through multi-turn interactions:
1. **Simulate Interactions**: Agent interacts with user simulator to write code
2. **Evaluate Code**: Test generated code against test cases
3. **Compute Metrics**: Calculate pass@k success rate

## Quick Start

### 1. VLLM Model Evaluation

```bash
# 1. Start user simulator server (in another terminal)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8001 \
    --tensor-parallel-size 8

# 2. Edit eval_vllm.sh
vim eval_vllm.sh
# Update:
#   AGENT_MODEL_PATH="/path/to/your/model"
#   USER_SIMULATOR_HOST="localhost:8001"

# 3. Run evaluation
./eval_vllm.sh
```

### 2. API Model Evaluation

```bash
# 1. Start user simulator server (in another terminal)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8001 \
    --tensor-parallel-size 8

# 2. Edit eval_api.sh
vim eval_api.sh
# Update:
#   AGENT_MODEL="gpt-4o"
#   OPENAI_API_KEY and OPENAI_BASE_URL

# 3. Run evaluation
export OPENAI_API_KEY="your-api-key"
./eval_api.sh
```

## Architecture

ColBench requires two models:
- **Agent Model**: Your trained model (VLLM) or API model (GPT-4, etc.)
- **User Simulator**: Simulates human user providing feedback (typically Llama-3.1-70B)

Both models work together in a multi-turn dialogue to solve coding tasks.

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `NUM_TASKS` | Number of tasks to evaluate | 100 |
| `BATCH_SIZE` | Parallel environments | 32 |
| `MAX_STEPS` | Maximum turns per task | 10 |
| `BEST_OF_N` | Samples per task (for Pass@k) | 1 |
| `TEMPERATURE` | Agent temperature | 1.0 |

### Data

Evaluation uses test split from ColBench:
```bash
data/colbench_code/test.parquet
```

## Outputs

Results are saved to `outputs/colbench/<experiment_name>/`:

```
outputs/colbench/
└── my_model_test/
    ├── trajectories.jsonl          # Dialogue histories and answers
    └── evaluated_trajectories.jsonl # With rewards added
```

### Output Metrics

After evaluation, you'll see:
```
Average correctness: 0.75
Number of trajectories: 100
Percentage of correct trajectories: 0.65
Best-of-k Average correctness: 0.80
```

Metrics:
- **Average correctness**: Average code correctness across all test cases
- **Percentage correct**: Percentage of fully correct solutions (1.0)
- **Best-of-k**: Success rate when taking best of k samples

## Evaluation Process

### Step 1: Generate Trajectories

```bash
python run_simulate.py \
    --agent_model /path/to/model \
    --user_simulator_host localhost:8001 \
    --output_path outputs/colbench/experiment/trajectories.jsonl
```

This runs the agent-user interaction and saves dialogue histories.

### Step 2: Evaluate Code

```bash
python run_evaluate.py \
    --saved_path outputs/colbench/experiment/trajectories.jsonl \
    --k 1
```

This:
1. Extracts generated code from trajectories
2. Runs code against test cases
3. Computes correctness metrics
4. Saves rewards back to the JSONL file

## Code Evaluation

ColBench evaluates code by:
1. **Extracting** the final code from agent's answer
2. **Executing** code against test cases from the dataset
3. **Comparing** outputs with ground truth
4. **Computing** pass rate across all test cases

Safety measures:
- Timeout protection (1 second per test case)
- Blacklist dangerous operations (os, sys, file I/O)
- Subprocess isolation

## Data Format

### Input (test.parquet)

Each task contains:
```json
{
  "problem_description": "Write a function that...",
  "ground_truth": "def solution(x):\n    return x * 2",
  "test_cases": {
    "test_1": "solution(5)",
    "test_2": "solution(10)"
  }
}
```

### Output (trajectories.jsonl)

Each trajectory contains:
```json
{
  "task": {...},
  "dialogue_history": [
    {"input": "...", "output": "..."},
    ...
  ],
  "answer": "def solution(x):\n    return x * 2",
  "reward": 1.0
}
```

## Troubleshooting

### User Simulator Server Issues

```bash
# Check if server is running
curl http://localhost:8001/v1/models

# Check GPU memory
nvidia-smi
```

### Import Errors

```bash
# Install dependencies
pip install fire vllm transformers datasets openai
pip install -e ../../sweet_rl  # Install sweet_rl package
```

### Code Execution Timeout

If code execution is slow:
- Reduce `NUM_TASKS` for testing
- Check for infinite loops in generated code
- Adjust timeout in `code_utils.py`

## Comparison with Training

Training uses the same environment:
- **Training**: ColBenchGym wraps sweet_rl environment
- **Evaluation**: Direct sweet_rl scripts

Key alignment points:
- Same prompts (agent_prompt, user_prompt)
- Same max_steps (10)
- Same user simulator model
- Same code evaluation logic

## Notes

- User simulator typically needs a strong model (Llama-3.1-70B or better)
- Agent can be smaller (Llama-3.1-8B, Qwen-2.5-7B, etc.)
- Evaluation is compute-intensive due to code execution
- Best-of-k requires k samples per task (set `BEST_OF_N=k`)
