"""
Test script for ColBenchGym environments.
This script demonstrates basic usage and verifies the environments work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from colbenchgym import ColBenchCodeEnv, get_code_config


def test_code_env():
    """Test the Backend Programming environment."""
    print("=" * 60)
    print("Testing ColBenchCodeEnv (Backend Programming)")
    print("=" * 60)

    # Create config
    config = get_code_config()
    config.env_hostname = "localhost"
    config.env_port = 8000
    config.env_model_name = "meta-llama/Llama-3.1-70B-Instruct"
    config.verbose = True
    config.max_steps = 3  # Shorter for testing

    # Create environment
    env = ColBenchCodeEnv(config=config)

    # Define test task
    task = {
        "problem_description": "Write a function to calculate factorial of a number",
        "ground_truth": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    }

    print("\nResetting environment...")
    obs, info = env.reset(options={"task": task})
    print(f"Initial observation: {obs['last_message']}")
    print(f"Step count: {obs['step_count']}")

    print("\n" + "-" * 60)
    print("Step 1: Ask clarification question")
    print("-" * 60)
    action = "Should the factorial function handle negative numbers? What should it return in that case?"
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Human response: {obs['last_message']}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if not (terminated or truncated):
        print("\n" + "-" * 60)
        print("Step 2: Provide solution")
        print("-" * 60)
        action = "I WANT TO ANSWER:\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Final answer: {info['agent_answer'][:100]}...")
        print(f"Terminated: {terminated}")

    env.close()
    print("\n✅ ColBenchCodeEnv test completed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ColBenchGym Test Suite")
    print("=" * 60)
    print("\nNOTE: These tests require a VLLM server running on localhost:8000")
    print("Start the server before running these tests:")
    print("\nFor code tasks:")
    print("  python -m vllm.entrypoints.openai.api_server \\")
    print("    --model meta-llama/Llama-3.1-70B-Instruct \\")
    print("    --port 8000 --max-model-len 16384")
    print("\n" + "=" * 60)

    try:
        # Test code environment
        test_code_env()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
