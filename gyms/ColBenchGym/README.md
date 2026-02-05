# ColBenchGym

A Gymnasium-compatible environment for Collaborative Agent Bench (ColBench) tasks, enabling reinforcement learning research on multi-turn collaborative interactions between LLM agents and human collaborators.

## Overview

ColBenchGym provides two types of collaborative environments:

1. **Backend Programming (`ColBenchCodeEnv`)**: Agents help users solve Python programming problems through clarification questions and iterative refinement.
2. **Frontend Design (`ColBenchHtmlEnv`)**: Agents create HTML/CSS websites through visual feedback and iterative design improvements.

## Features

- **Multi-turn Interaction**: Supports up to 10 rounds of back-and-forth conversation
- **Human Simulation**: Uses VLLM server to simulate human collaborator responses
- **Gymnasium Interface**: Full compatibility with standard RL training pipelines
- **Context Consistency**: Maintains the same agent context as the original sweet-rl implementation
- **Flexible Configuration**: Dataclass-based configuration with YAML support

## Installation

```bash
cd gyms/ColBenchGym
pip install -e .
```

For Frontend Design tasks, you also need Firefox and GeckoDriver:

```bash
# Install Firefox (or download from https://www.mozilla.org/firefox/)
# Install GeckoDriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xvzf geckodriver-v0.35.0-linux64.tar.gz
sudo mv geckodriver /usr/local/bin/
```

## Quick Start

### Backend Programming

```python
from colbenchgym import ColBenchCodeEnv, get_code_config

# Configure environment
config = get_code_config()
config.env_hostname = "localhost"  # VLLM server hostname
config.env_port = 8000  # VLLM server port
config.env_model_name = "meta-llama/Llama-3.1-70B-Instruct"
config.verbose = True

# Create environment
env = ColBenchCodeEnv(config=config)

# Define a task
task = {
    "problem_description": "Help me write a function to check if a number is prime",
    "ground_truth": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
}

# Reset environment
obs, info = env.reset(options={"task": task})

# Interact with environment
action = "What are the requirements for the prime checking function? Should it handle negative numbers?"
obs, reward, terminated, truncated, info = env.step(action)

print(f"Human response: {obs['last_message']}")
print(f"Dialogue history:\n{obs['dialogue_history']}")
```

### Frontend Design

```python
from colbenchgym import ColBenchHtmlEnv, get_html_config

# Configure environment
config = get_html_config()
config.env_hostname = "localhost"
config.env_port = 8000
config.env_model_name = "Qwen/Qwen2-VL-72B-Instruct"
config.verbose = True

# Create environment
env = ColBenchHtmlEnv(config=config)

# Define a task
task = {
    "problem_description": "Create a landing page for a coffee shop",
    "ground_truth": "<html>...</html>"  # Ground truth HTML
}

# Reset and interact
obs, info = env.reset(options={"task": task})
action = "OUTPUT:\n<html><body><h1>Coffee Shop</h1></body></html>"
obs, reward, terminated, truncated, info = env.step(action)
```

## Data Preprocessing

### Download ColBench Data

```bash
huggingface-cli download facebook/collaborative_agent_bench \
    backend_tasks/train.jsonl \
    backend_tasks/test.jsonl \
    frontend_tasks/train.jsonl \
    frontend_tasks/test.jsonl
```

### Preprocess for UserRL Training

**Backend Programming:**
```bash
python examples/data_preprocess/colbench_code_multiturn.py \
    --train_data /path/to/backend_tasks/train.jsonl \
    --test_data /path/to/backend_tasks/test.jsonl \
    --local_dir ./data/colbench_code \
    --train_size 1000 \
    --test_size 100
```

**Frontend Design:**
```bash
python examples/data_preprocess/colbench_html_multiturn.py \
    --train_data /path/to/frontend_tasks/train.jsonl \
    --test_data /path/to/frontend_tasks/test.jsonl \
    --local_dir ./data/colbench_html \
    --train_size 500 \
    --test_size 50
```

## Setting Up VLLM Server

The environments require a VLLM server to simulate human collaborator responses.

### For Backend Programming:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --max-model-len 16384 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --port 8000 \
    --enforce-eager \
    --trust-remote-code
```

### For Frontend Design:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-72B-Instruct \
    --max-model-len 16384 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --port 8000 \
    --enforce-eager \
    --limit-mm-per-prompt image=2 \
    --trust-remote-code
```

## Configuration

### ColBenchGymConfig

```python
from colbenchgym import ColBenchGymConfig

config = ColBenchGymConfig(
    task_type="code",  # or "html"
    env_hostname="localhost",
    env_port=8000,
    env_model_name="meta-llama/Llama-3.1-70B-Instruct",
    max_steps=10,
    verbose=True,
    seed=42,
    human_response_char_limit=400,  # for code tasks
    human_response_char_limit_html=1000,  # for html tasks
    reward_scale=1.0,
    step_penalty=0.0,
)
```

## Integration with UserRL

The environments are designed to work seamlessly with the UserRL training pipeline:

1. **Data Format**: Preprocessed data follows UserRL's parquet format with `interact_with_env` tool
2. **Context Consistency**: Agent prompts match the original sweet-rl implementation
3. **Reward Model**: Compatible with UserRL's reward model interface
4. **Tool Integration**: Supports UserRL's tool-based interaction pattern

## Action Format

Both environments use text actions:

### Backend Programming:
- Regular response: `"Can you clarify the input format?"`
- Final answer: `"I WANT TO ANSWER:\ndef is_prime(n):\n    ..."`

### Frontend Design:
- Design proposal: `"OUTPUT:\n<html>...</html>"`
- Final answer: `"I WANT TO ANSWER:\n<html>...</html>"`

## Observation Format

```python
{
    "dialogue_history": str,  # Full conversation history
    "step_count": int,  # Current step number
    "episode_complete": bool,  # Whether episode is done
    "last_message": str  # Most recent message from environment
}
```

## Info Dictionary

```python
{
    "problem_description": str,
    "ground_truth": str,
    "agent_answer": str or path,  # Code string or image path
    "task_id": str,
    "category": str,
    "dialogue_messages": List[Dict],  # Structured dialogue
    "raw_agent_response": str
}
```

## Evaluation

Evaluation is done externally (similar to sweet-rl):

**Code Tasks:**
```bash
# Use sweet_rl's evaluate_code.py or implement custom evaluation
python sweet_rl/scripts/evaluate_code.py /path/to/outputs.jsonl
```

**HTML Tasks:**
```bash
# Use sweet_rl's evaluate_html.py or implement custom evaluation
python sweet_rl/scripts/evaluate_html.py /path/to/outputs.jsonl
```

## Key Differences from sweet-rl

1. **Interface**: Gymnasium API vs. custom environment API
2. **Batching**: Single environment instance (batch via vectorized envs)
3. **Reward**: Zero reward during interaction, computed externally
4. **State**: Observable state vs. internal state management

## Citation

If you use ColBenchGym in your research, please cite the original ColBench paper:

```bibtex
@misc{zhou2025sweetrltrainingmultiturnllm,
    title={SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks},
    author={Yifei Zhou and Song Jiang and Yuandong Tian and Jason Weston and Sergey Levine and Sainbayar Sukhbaatar and Xian Li},
    year={2025},
    eprint={2503.15478},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2503.15478},
}
```

## License

This implementation follows the CC-By-NC license of the original ColBench/sweet-rl code.
