# Tau2Gym - Tau2-Bench Integration for UserRL

A Gymnasium-compatible environment that integrates [tau2-bench](https://github.com/sierra-research/tau2-bench) into the UserRL framework for reinforcement learning training of conversational agents.

## Overview

Tau2Gym provides a seamless integration between tau2-bench and UserRL by wrapping tau2-bench's `AgentGymEnv` while preserving all original functionality:

- **Original Prompts**: All tau2-bench prompts and agent context are preserved exactly
- **Original Evaluation**: Uses tau2-bench's evaluation logic for fair comparison
- **Train/Test Splits**: Proper data splits for RL training and evaluation
- **Multiple Domains**: Support for retail, airline, and telecom domains
- **Solo & Interactive Modes**: Both autonomous and user-interactive modes

## Key Design Principles

1. **Minimal Modification**: Direct wrapper around tau2-bench's AgentGymEnv
2. **Identical Results**: Testing on tau2-bench should produce identical results
3. **Proper Splits**: Support for train/test/base splits for RL experiments
4. **UserRL Compatible**: Integrates seamlessly with UserRL's training pipeline

## Installation

### Prerequisites

- Python 3.10+
- tau2-bench installed from source

### Step 1: Install tau2-bench

```bash
cd /path/to/UserRL/tau2-bench
pip install -e .
```

### Step 2: Install Tau2Gym

```bash
cd /path/to/UserRL/gyms/Tau2Gym
pip install -e .
```

## Usage

### Basic Usage

```python
import tau2gym
from tau2gym import Tau2Env, get_train_config

# Create environment for training
config = get_train_config(domain="retail")
env = Tau2Env(config)

# Reset environment
observation, info = env.reset()
print(f"Initial observation: {observation}")

# Step through episode
action = "Hello! I'd be happy to help you today."
observation, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Done: {terminated or truncated}")

# Clean up
env.close()
```

### Configuration Options

```python
from tau2gym import Tau2GymConfig

config = Tau2GymConfig(
    domain="retail",          # Domain: retail, airline, telecom, mock
    task_split="train",       # Split: train, test, or base (full dataset)
    data_mode="random",       # How to select tasks: random, sequential, single
    max_steps=30,             # Maximum steps per episode
    verbose=True,             # Print detailed information
    solo_mode=False,          # True for autonomous agent, False for user interaction
    user_llm="gpt-4o",       # LLM for user simulator
    user_temperature=0.7,     # Temperature for user simulator
)

env = Tau2Env(config)
```

### Train/Test Splits

Tau2Gym supports proper train/test splits for RL experiments:

```python
from tau2gym import get_train_config, get_test_config

# Training configuration
train_config = get_train_config(domain="retail")
train_env = Tau2Env(train_config)

# Testing configuration
test_config = get_test_config(domain="retail")
test_env = Tau2Env(test_config)

# For comparison with original tau2-bench results, use "base" split
from tau2gym import Tau2GymConfig
eval_config = Tau2GymConfig(domain="retail", task_split="base")
eval_env = Tau2Env(eval_config)
```

**Important**:
- Use `task_split="train"` for training your RL agent
- Use `task_split="test"` for evaluating your trained agent
- Use `task_split="base"` only when comparing with original tau2-bench results

### Available Domains

- `retail`: Retail customer service scenarios
- `airline`: Airline booking and support scenarios
- `telecom`: Telecom troubleshooting and support scenarios
- `mock`: Simple mock domain for testing

### Action Format

Tau2Gym uses tau2-bench's action format:

**Message to User:**
```python
action = "I'd be happy to help you with that!"
```

**Tool Call (Functional Format):**
```python
action = "search_products(category='electronics', price_max=500)"
```

**Tool Call (JSON Format):**
```python
action = '{"name": "search_products", "arguments": {"category": "electronics", "price_max": 500}}'
```

### Observation Format

Observations are strings containing the conversation history in tau2-bench's format:

```
user: I need to return a product
assistant: I can help you with that. Let me look up your recent orders.
assistant: search_orders(user_id='user_123')
tool: {"name": "search_orders", "result": "Found 3 recent orders..."}
```

## Integration with UserRL Training

### Data Preprocessing

Create a data preprocessing script to prepare tau2-bench data for UserRL:

```python
# See examples/data_preprocess/tau2_multiturn_w_tool.py for complete example

import json
from tau2gym import Tau2Env, get_train_config

def create_tau2_dataset(domain: str, split: str, output_file: str):
    """Create UserRL-compatible dataset from tau2-bench."""
    config = get_train_config(domain=domain)
    config.task_split = split

    env = Tau2Env(config)

    # Generate prompts for each task
    dataset = []
    for task_id in env.task_ids:
        env.config.data_source = task_id
        observation, info = env.reset()

        dataset.append({
            "task_id": task_id,
            "prompt": observation,
            "domain": domain,
            "policy": info.get("policy", ""),
            "tools": [str(tool) for tool in info.get("tools", [])],
        })

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

# Create training dataset
create_tau2_dataset("retail", "train", "tau2_retail_train.json")
```

### Training Configuration

Configure UserRL to use Tau2Gym (see `examples/tau2/train.sh`):

```yaml
# config/tau2_trainer.yaml
rollout:
  name: tau2gym
  rollout_config:
    domain: retail
    task_split: train
    max_steps: 30

actor_rollout_ref:
  env_type: "tau2gym"
  env_config:
    domain: retail
    task_split: train
```

## Verification

### Test Script

Verify that Tau2Gym produces identical results to tau2-bench:

```python
# test_verification.py
import gymnasium as gym
from tau2gym import Tau2Env, Tau2GymConfig
from tau2.gym.gym_agent import AgentGymEnv, TAU_BENCH_ENV_ID

def test_identical_results():
    """Test that Tau2Gym produces identical results to tau2-bench."""
    task_id = "retail_task_1"
    domain = "retail"

    # Create tau2-bench environment directly
    tau2_env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task_id)
    tau2_obs, tau2_info = tau2_env.reset()

    # Create Tau2Gym environment
    config = Tau2GymConfig(domain=domain, data_source=task_id, task_split="base")
    tau2gym_env = Tau2Env(config)
    gym_obs, gym_info = tau2gym_env.reset()

    # Verify observations match
    assert tau2_obs == gym_obs, "Observations don't match!"

    # Test a few steps
    test_action = "Hello! How can I help you today?"

    tau2_obs, tau2_reward, tau2_term, tau2_trunc, tau2_info = tau2_env.step(test_action)
    gym_obs, gym_reward, gym_term, gym_trunc, gym_info = tau2gym_env.step(test_action)

    assert tau2_obs == gym_obs, "Step observations don't match!"
    assert tau2_reward == gym_reward, "Rewards don't match!"

    print("✅ Verification passed! Tau2Gym produces identical results to tau2-bench.")

if __name__ == "__main__":
    test_identical_results()
```

## Data Splits

Tau2Gym respects tau2-bench's train/test splits defined in `split_tasks.json`:

```json
{
  "train": ["task_1", "task_2", "task_3"],
  "test": ["task_4", "task_5"]
}
```

The splits are loaded automatically based on the `task_split` configuration.

## Reward Structure

Tau2Gym uses tau2-bench's original reward computation:

- **Action-based**: Rewards for completing required actions correctly
- **NL Assertions**: Rewards for satisfying natural language criteria
- **Env Assertions**: Rewards for achieving correct environment states

Rewards can be scaled and normalized via configuration:

```python
config = Tau2GymConfig(
    reward_scale=1.0,        # Scale factor for rewards
    step_penalty=0.01,       # Penalty per step (encourages efficiency)
    normalize_rewards=True,  # Normalize rewards to [0, 1]
)
```

## Troubleshooting

### tau2-bench not found

```
ImportError: tau2-bench is not installed.
```

**Solution**: Install tau2-bench from source:
```bash
cd /path/to/UserRL/tau2-bench
pip install -e .
```

### TAU2_DATA_DIR not set

```
ValueError: tau2_data_dir not found
```

**Solution**: Set the TAU2_DATA_DIR environment variable:
```bash
export TAU2_DATA_DIR=/path/to/UserRL/tau2-bench/data
```

Or provide it in config:
```python
config = Tau2GymConfig(tau2_data_dir="/path/to/tau2-bench/data")
```

### No tasks found for split

```
ValueError: No task IDs available.
```

**Solution**: Ensure `split_tasks.json` exists in the domain's data directory, or use `task_split="base"` to load all tasks.

## Architecture

```
Tau2Gym
├── tau2gym/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration classes
│   └── env/
│       ├── __init__.py
│       └── tau2_env.py       # Main environment (wraps AgentGymEnv)
├── README.md
├── setup.py
├── requirements.txt
└── test_verification.py      # Verification tests
```

## Contributing

When contributing to Tau2Gym:

1. **Preserve tau2-bench behavior**: Any changes must maintain identical results to tau2-bench
2. **Test verification**: Run verification tests to ensure compatibility
3. **Document changes**: Update README with any new features or changes
4. **Follow UserRL patterns**: Match the structure and style of other gyms

## References

- [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [tau2-bench Paper](https://arxiv.org/abs/2506.07982)
- [UserRL Documentation](../../README.md)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Citation

If you use Tau2Gym in your research, please cite both tau2-bench and UserRL:

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment},
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
}
```
