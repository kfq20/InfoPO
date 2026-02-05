# ColBench Migration Guide

## Overview

ColBenchGym successfully migrates the Collaborative Agent Bench from sweet_rl to UserRL, providing two multi-turn collaborative environments:

1. **ColBenchCodeEnv**: Backend Programming tasks
2. **ColBenchHtmlEnv**: Frontend Design tasks

## Key Features

✅ **Environment Consistency**: Maintains the same agent context as sweet_rl
✅ **Gymnasium Interface**: Full compatibility with standard RL pipelines
✅ **Data Compatibility**: Works with UserRL's training framework
✅ **Tool Integration**: Supports UserRL's `interact_with_env` tool pattern

## Directory Structure

```
gyms/ColBenchGym/
├── colbenchgym/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclass
│   ├── utils.py               # HTML rendering utilities
│   ├── env/
│   │   ├── __init__.py
│   │   ├── code_env.py        # Backend Programming environment
│   │   └── html_env.py        # Frontend Design environment
│   └── prompts/
│       ├── llm_agent_code_prompt.txt
│       ├── human_simulator_code_prompt.txt
│       ├── llm_agent_html_prompt.txt
│       └── human_simulator_html_prompt.txt
├── setup.py
├── README.md
└── test_colbench.py           # Test script

examples/data_preprocess/
├── colbench_code_multiturn.py # Code data preprocessing
└── colbench_html_multiturn.py # HTML data preprocessing
```

## Installation

### 1. Install ColBenchGym

```bash
cd gyms/ColBenchGym
pip install -e .
```

### 2. Install Dependencies for HTML Tasks (Optional)

```bash
# Install Firefox
# Linux:
sudo apt-get install firefox
# macOS:
brew install --cask firefox

# Install GeckoDriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xvzf geckodriver-v0.35.0-linux64.tar.gz
sudo mv geckodriver /usr/local/bin/
```

## Usage

### Quick Start

```python
from colbenchgym import ColBenchCodeEnv, get_code_config

# Configure
config = get_code_config()
config.env_hostname = "localhost"
config.env_port = 8000
config.env_model_name = "meta-llama/Llama-3.1-70B-Instruct"

# Create environment
env = ColBenchCodeEnv(config=config)

# Define task
task = {
    "problem_description": "Write a function to check if a number is prime",
    "ground_truth": "def is_prime(n):..."
}

# Interact
obs, info = env.reset(options={"task": task})
obs, reward, terminated, truncated, info = env.step(
    "What are the requirements for the prime checking function?"
)
```

### Data Preprocessing

Download ColBench data:
```bash
huggingface-cli download facebook/collaborative_agent_bench \
    backend_tasks/train.jsonl \
    backend_tasks/test.jsonl
```

Preprocess for UserRL:
```bash
python examples/data_preprocess/colbench_code_multiturn.py \
    --train_data /path/to/backend_tasks/train.jsonl \
    --test_data /path/to/backend_tasks/test.jsonl \
    --local_dir ./data/colbench_code
```

### VLLM Server Setup

Start VLLM server for human simulation:

```bash
# For code tasks
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --max-model-len 16384 \
    --tensor-parallel-size 8 \
    --port 8000
```

## Agent Context Consistency

The agent context in ColBenchGym matches sweet_rl exactly:

### Backend Programming (Code)

**System Prompt (from `llm_agent_code_prompt.txt`):**
```
You are a helpful LLM agent.
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information
by asking questions to the human user about some hidden information and implicit constraints.
...
```

**Dialogue Format:**
```
[
  {"role": "system", "content": agent_prompt},
  {"role": "user", "content": problem_description},
  {"role": "assistant", "content": agent_response_1},
  {"role": "user", "content": human_response_1},
  ...
]
```

### Frontend Design (HTML)

**System Prompt (from `llm_agent_html_prompt.txt`):**
```
You are a helpful LLM agent.
Your task is to help a human user to code a complete website with a good design in HTML and Tailwind CSS.
...
```

**Action Format:**
- First output thoughts
- Then say "OUTPUT:\n" followed by HTML code
- Human sees rendered image and provides feedback

## Integration with UserRL

### Data Format

Preprocessed data follows UserRL's format:

```python
{
    "data_source": "colbench_code",
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "ability": "interaction",
    "reward_model": {
        "style": "rule",
        "ground_truth": "...",
        "env_name": "ColBenchCodeEnv",
        ...
    },
    "extra_info": {
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "interact_with_env": {
                "create_kwargs": {
                    "env_name": "ColBenchCodeEnv",
                    "task": {...}
                }
            }
        }
    }
}
```

### Tool Integration

The environments work with UserRL's `interact_with_env` tool:

```python
# In UserRL training
tools_kwargs = data["extra_info"]["tools_kwargs"]
env = create_env_from_kwargs(
    env_name="ColBenchCodeEnv",
    **tools_kwargs["interact_with_env"]["create_kwargs"]
)
```

## Testing

Run the test script:

```bash
cd gyms/ColBenchGym
python test_colbench.py
```

Note: Requires VLLM server running on localhost:8000.

## Differences from sweet_rl

| Aspect | sweet_rl | ColBenchGym |
|--------|----------|-------------|
| Interface | Custom env API | Gymnasium API |
| Batching | Built-in parallel envs | Single env (use vectorized envs) |
| Agent | VLLMAgent class | External (UserRL handles) |
| Reward | Computed externally | Zero during interaction |
| Reset | `reset(problem, ground_truth)` | `reset(options={"task": {...}})` |
| Step | `step(response, formatted_prompt)` | `step(action_string)` |
| Dialogue | Internal state | Observable in `obs` |

## Next Steps

1. **Download Data**: Get ColBench data from HuggingFace
2. **Preprocess**: Run data preprocessing scripts
3. **Setup VLLM**: Start VLLM server for human simulation
4. **Train**: Use UserRL's training pipeline with preprocessed data
5. **Evaluate**: Run evaluation using sweet_rl's evaluation scripts

## Troubleshooting

### VLLM Server Connection Issues
```python
config.env_hostname = "localhost"  # or your server IP
config.env_port = 8000
config.env_base_url = "http://localhost:8000/v1"  # explicit URL
```

### Firefox/GeckoDriver Not Found
```bash
# Check if installed
geckodriver --version
firefox --version

# Add to PATH if needed
export PATH=$PATH:/path/to/geckodriver
```

### Import Errors
```bash
# Reinstall in editable mode
cd gyms/ColBenchGym
pip install -e .
```

## Citation

If you use ColBenchGym, please cite the original ColBench paper:

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
