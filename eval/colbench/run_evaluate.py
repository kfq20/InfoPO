#!/usr/bin/env python3
"""
Simplified ColBench code evaluation script.
Wraps sweet_rl's evaluate_code.py.
"""
import sys
from pathlib import Path
from fire import Fire

# Add sweet_rl directory to path
SWEET_RL_PARENT_DIR = Path(__file__).parent.parent.parent.parent
SWEET_RL_DIR = SWEET_RL_PARENT_DIR / "sweet_rl"
# Add both parent directory (for sweet_rl package) and sweet_rl directory itself
if str(SWEET_RL_DIR) not in sys.path:
    sys.path.insert(0, str(SWEET_RL_DIR))
if str(SWEET_RL_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(SWEET_RL_PARENT_DIR))

from sweet_rl.scripts.evaluate_code import main as evaluate_main

def run_code_evaluation(
    saved_path: str,
    k: int = 1,
):
    """
    Evaluate generated code from trajectories.

    Args:
        saved_path: Path to trajectories.jsonl file
        k: Number of samples per task (for Best-of-k)
    """
    print("=" * 80)
    print("ColBench Code Evaluation")
    print("=" * 80)
    print(f"Trajectories: {saved_path}")
    print(f"Best-of-{k} evaluation")
    print("=" * 80)
    print()

    # Call sweet_rl's evaluate_code
    evaluate_main(saved_path=saved_path, k=k)

    print()
    print("=" * 80)
    print("Evaluation complete!")
    print(f"Rewards saved to: {saved_path}")
    print("=" * 80)

if __name__ == "__main__":
    Fire(run_code_evaluation)
