# ColBench Training with UserRL

This directory contains the training configuration and scripts for ColBench (Collaborative Agent Benchmark) code collaboration tasks using the UserRL framework.

## Overview

ColBench is a benchmark that evaluates agents' ability to collaborate with humans on programming tasks. The key challenge is understanding implicit requirements through multi-turn interaction.

## Directory Structure

```
examples/colbench/
├── config/
│   ├── colbench_trainer.yaml          # Main training configuration
│   └── tool_config/
│       └── colbench_tool_config.yaml   # Tool configuration for interaction
├── train.sh                            # Training script
└── README.md                           # This file
```

## Setup

### 1. Install ColBenchGym

```bash
pip install -e gyms/ColBenchGym
```

### 2. Prepare Data

The training data should be in the `data/colbench_code/` directory:
- `train.parquet`: Training data
- `test.parquet`: Validation data

### 3. Configure Environment Variables

Edit `train.sh` to set:

```bash
# Model path
MODEL_PATH="/path/to/your/model"

# OpenAI API for human simulation
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_API_BASE="https://api.openai.com/v1"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=8
```

### 4. (Optional) Setup VLLM Server for Human Simulation

If you want to use a local VLLM server instead of OpenAI API:

```bash
export COLBENCH_ENV_HOSTNAME="localhost"
export COLBENCH_ENV_PORT="8000"
export COLBENCH_ENV_MODEL_NAME="your-model-name"
```

## Training

### Quick Start

```bash
cd examples/colbench
./train.sh
```

### Key Configuration Parameters

#### Data Configuration
- `data.train_batch_size=128`: Training batch size
- `data.max_prompt_length=2048`: Maximum prompt length
- `data.max_response_length=8192`: Maximum response length

#### Multi-turn Interaction
- `actor_rollout_ref.rollout.multi_turn.enable=True`: Enable multi-turn interaction
- `actor_rollout_ref.rollout.multi_turn.max_turns=10`: Maximum interaction turns
- `actor_rollout_ref.rollout.multi_turn.turn_level_method="Equalized"`: Turn-level credit assignment
- `actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum"`: Trajectory scoring method

#### GRPO Algorithm
- `algorithm.adv_estimator=grpo_multiturn`: Use GRPO with multi-turn support
- `algorithm.gamma=0.8`: Discount factor
- `algorithm.action_credit_ratio=0.8`: Credit assignment ratio between actions and observations

#### Model Configuration
- `actor_rollout_ref.rollout.n=4`: Number of rollouts per prompt (for GRPO)
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`: Tensor parallelism size
- `actor_rollout_ref.actor.optim.lr=1e-6`: Learning rate

## Task Format

ColBench tasks have the following structure:

```python
{
    'data_source': 'colbench_code',
    'prompt': [  # Multi-turn conversation format
        {'role': 'system', 'content': '...'},
        {'role': 'user', 'content': '...'}
    ],
    'reward_model': {
        'env_name': 'ColBenchCodeEnv',
        'ground_truth': '...',  # Ground truth code
        'problem_description': '...',
        'style': 'rule'
    }
}
```

## Interaction Protocol

The agent interacts with the environment using the `interact_with_env` tool:

```python
{
    "choice": "action",
    "content": "Your response or question"
}
```

To provide the final answer:
```
I WANT TO ANSWER:
def your_solution():
    # Your code here
    pass
```

## Evaluation

The agent is evaluated based on:
1. **Code Correctness**: Whether the final solution passes test cases
2. **Interaction Efficiency**: Number of turns used to gather information
3. **Question Quality**: Relevance and effectiveness of clarification questions

## Advanced Configuration

### Custom Learning Rate Schedule

Override in training script:
```bash
actor_rollout_ref.actor.optim.lr=5e-7
```

### Adjust Rollout Settings

For more diverse rollouts:
```bash
actor_rollout_ref.rollout.n=8
```

### Memory Optimization

For limited GPU memory:
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
```

## Monitoring

Training logs are saved to:
- WandB project: `ColBench-UserRL`
- Experiment name: `colbench_code_training`
- Local logs: `outputs/colbench_training/`
- Evaluation trajectories: `outputs/colbench_training/eval_logs/`

## Troubleshooting

### OOM (Out of Memory)
- Reduce `data.train_batch_size`
- Reduce `actor_rollout_ref.rollout.gpu_memory_utilization`
- Reduce `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- Increase `actor_rollout_ref.rollout.tensor_model_parallel_size`

### Slow Training
- Check if `max_turns` is too high
- Verify human simulator (API) response time
- Consider using local VLLM server instead of API

### API Rate Limiting
- Use a local VLLM server for human simulation
- Adjust request rate in environment configuration

## References

- [ColBench Paper](https://arxiv.org/abs/2405.12195)
- [ColBenchGym Documentation](../../gyms/ColBenchGym/README.md)
- [UserRL Framework](../../README.md)

## Citation

If you use ColBench in your research, please cite:

```bibtex
@article{colbench2024,
  title={ColBench: A Benchmark for Evaluating Collaborative Agents},
  author={...},
  year={2024}
}
```
