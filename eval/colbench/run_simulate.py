#!/usr/bin/env python3
"""
Simplified ColBench evaluation script for VLLM models.
Wraps sweet_rl's simulate_interactions.py with environment setup.
"""
import os
import sys
import json
from pathlib import Path
from fire import Fire

# Add sweet_rl to path
SWEET_RL_DIR = Path(__file__).parent.parent.parent / "sweet_rl"
sys.path.insert(0, str(SWEET_RL_DIR))

from scripts.simulate_interactions import main as simulate_main

def run_vllm_evaluation(
    agent_model_path: str,
    user_simulator_host: str = "localhost:8001",
    output_dir: str = "outputs/colbench",
    experiment_name: str = "vllm_test",
    num_tasks: int = 100,
    batch_size: int = 32,
    max_steps: int = 10,
    best_of_n: int = 1,
    temperature: float = 1.0,
    task_type: str = "code",
):
    """
    Run ColBench evaluation for VLLM-hosted model.

    Args:
        agent_model_path: Path to agent model
        user_simulator_host: Host:port for user simulator VLLM server
        output_dir: Output directory
        experiment_name: Experiment name
        num_tasks: Number of tasks to evaluate
        batch_size: Batch size for parallel execution
        max_steps: Max conversation turns
        best_of_n: Number of samples per task
        temperature: Sampling temperature
        task_type: Task type (code or html)
    """
    # Setup paths
    output_dir = Path(output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectories.jsonl"

    # Get data path
    data_dir = Path(__file__).parent.parent.parent / "data" / "colbench_code"
    input_path = data_dir / "test.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Data not found at {input_path}")

    # Get prompt paths
    prompt_dir = SWEET_RL_DIR / "prompts"
    if task_type == "code":
        user_prompt_path = prompt_dir / "human_simulator_code_prompt.txt"
        agent_prompt_path = prompt_dir / "llm_agent_code_prompt.txt"
    else:
        user_prompt_path = prompt_dir / "human_simulator_html_prompt.txt"
        agent_prompt_path = prompt_dir / "llm_agent_html_prompt.txt"

    # Environment model (user simulator)
    env_model = "meta-llama/Llama-3.1-70B-Instruct"  # Default
    user_sim_port = int(user_simulator_host.split(":")[-1])
    user_sim_hostname = user_simulator_host.split(":")[0]

    print("=" * 80)
    print("ColBench Evaluation - VLLM Model")
    print("=" * 80)
    print(f"Agent Model: {agent_model_path}")
    print(f"User Simulator: {user_simulator_host}")
    print(f"Num Tasks: {num_tasks}")
    print(f"Best-of-{best_of_n}")
    print(f"Output: {output_path}")
    print("=" * 80)
    print()

    # Call sweet_rl's simulate_interactions
    simulate_main(
        hostname=user_sim_hostname,
        input_path=str(input_path),
        output_path=str(output_path),
        user_prompt_path=str(user_prompt_path),
        agent_prompt_path=str(agent_prompt_path),
        agent_model=agent_model_path,
        env_model=env_model,
        batch_size=batch_size,
        num_tasks=num_tasks,
        max_steps=max_steps,
        best_of_n=best_of_n,
        task_type=task_type,
        port=user_sim_port,
        temperature=temperature,
    )

    print()
    print("=" * 80)
    print("Trajectory generation complete!")
    print(f"Saved to: {output_path}")
    print()
    print("Next step: Run evaluation")
    print(f"  python run_evaluate.py --saved_path {output_path} --k {best_of_n}")
    print("=" * 80)

if __name__ == "__main__":
    Fire(run_vllm_evaluation)
