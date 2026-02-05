#!/usr/bin/env python3
"""
Verification Test for Tau2Gym

This script verifies that Tau2Gym produces identical results to tau2-bench's
original AgentGymEnv. This ensures that we preserve all prompts, agent context,
and evaluation logic when integrating with UserRL.
"""

import sys
import traceback
from typing import List, Tuple

try:
    import gymnasium as gym
    from tau2.gym.gym_agent import AgentGymEnv, register_gym_agent, TAU_BENCH_ENV_ID
    from tau2gym import Tau2Env, Tau2GymConfig
    TAU2_AVAILABLE = True
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please ensure tau2-bench and tau2gym are installed:")
    print("  cd tau2-bench && pip install -e .")
    print("  cd gyms/Tau2Gym && pip install -e .")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg: str):
    """Print success message in green."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")


def print_error(msg: str):
    """Print error message in red."""
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")


def print_info(msg: str):
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.ENDC}")


def print_warning(msg: str):
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.ENDC}")


def test_environment_creation(domain: str = "mock") -> bool:
    """Test that both environments can be created successfully."""
    print_info(f"Testing environment creation for domain: {domain}")

    try:
        # Register tau2-bench environments
        register_gym_agent()

        # Create tau2-bench environment
        tau2_env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=domain,
            task_id=None,  # Will use first task
            solo_mode=False
        )
        print_success("Created tau2-bench AgentGymEnv")

        # Create Tau2Gym environment
        config = Tau2GymConfig(
            domain=domain,
            task_split="base",
            data_mode="single",
            verbose=False
        )
        tau2gym_env = Tau2Env(config)
        print_success("Created Tau2Gym environment")

        tau2_env.close()
        tau2gym_env.close()

        return True

    except Exception as e:
        print_error(f"Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_reset_identical(domain: str = "mock", task_id: str = None) -> bool:
    """Test that reset produces identical observations."""
    print_info(f"Testing reset() for domain: {domain}")

    try:
        register_gym_agent()

        # Get task_id if not provided
        if task_id is None:
            config = Tau2GymConfig(domain=domain, task_split="base")
            temp_env = Tau2Env(config)
            if temp_env.task_ids:
                task_id = temp_env.task_ids[0]
            else:
                print_warning(f"No tasks found for domain {domain}")
                return False
            temp_env.close()

        print_info(f"Using task_id: {task_id}")

        # Create tau2-bench environment
        tau2_env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=domain,
            task_id=task_id,
            solo_mode=False
        )
        tau2_obs, tau2_info = tau2_env.reset()

        # Create Tau2Gym environment
        config = Tau2GymConfig(
            domain=domain,
            task_split="base",
            data_source=task_id,
            verbose=False
        )
        gym_env = Tau2Env(config)
        gym_obs, gym_info = gym_env.reset()

        # Compare observations
        if tau2_obs == gym_obs:
            print_success("Reset observations are identical")
            observations_match = True
        else:
            print_error("Reset observations differ")
            print(f"  tau2-bench observation length: {len(tau2_obs)}")
            print(f"  Tau2Gym observation length: {len(gym_obs)}")
            if len(tau2_obs) > 0 and len(gym_obs) > 0:
                print(f"  First 200 chars of tau2-bench: {tau2_obs[:200]}")
                print(f"  First 200 chars of Tau2Gym: {gym_obs[:200]}")
            observations_match = False

        # Compare info dictionaries (policy and tools should be present)
        tau2_policy = tau2_info.get('policy', '')
        gym_policy = gym_info.get('policy', '')

        if tau2_policy == gym_policy:
            print_success("Policy information is identical")
            policy_match = True
        else:
            print_error("Policy information differs")
            policy_match = False

        tau2_env.close()
        gym_env.close()

        return observations_match and policy_match

    except Exception as e:
        print_error(f"Reset test failed: {e}")
        traceback.print_exc()
        return False


def test_step_identical(domain: str = "mock", task_id: str = None, num_steps: int = 3) -> bool:
    """Test that step produces identical results."""
    print_info(f"Testing step() for {num_steps} steps")

    try:
        register_gym_agent()

        # Get task_id if not provided
        if task_id is None:
            config = Tau2GymConfig(domain=domain, task_split="base")
            temp_env = Tau2Env(config)
            if temp_env.task_ids:
                task_id = temp_env.task_ids[0]
            else:
                print_warning(f"No tasks found for domain {domain}")
                return False
            temp_env.close()

        # Create environments
        tau2_env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=domain,
            task_id=task_id,
            solo_mode=False
        )
        tau2_obs, tau2_info = tau2_env.reset()

        config = Tau2GymConfig(
            domain=domain,
            task_split="base",
            data_source=task_id,
            verbose=False,
            reward_scale=1.0,  # No scaling
            step_penalty=0.0,  # No penalty
            normalize_rewards=False,  # No normalization
        )
        gym_env = Tau2Env(config)
        gym_obs, gym_info = gym_env.reset()

        # Test actions
        test_actions = [
            "Hello! How can I help you today?",
            "Let me check that for you.",
            "done()",
        ]

        all_match = True
        for i, action in enumerate(test_actions[:num_steps], 1):
            print_info(f"Step {i}: {action[:50]}...")

            # Step both environments
            tau2_obs, tau2_reward, tau2_term, tau2_trunc, tau2_info = tau2_env.step(action)
            gym_obs, gym_reward, gym_term, gym_trunc, gym_info = gym_env.step(action)

            # Compare results
            obs_match = (tau2_obs == gym_obs)
            reward_match = (abs(tau2_reward - gym_reward) < 1e-6)
            term_match = (tau2_term == gym_term)
            trunc_match = (tau2_trunc == gym_trunc)

            if obs_match and reward_match and term_match and trunc_match:
                print_success(f"  Step {i} results are identical")
            else:
                print_error(f"  Step {i} results differ:")
                if not obs_match:
                    print(f"    Observations differ")
                if not reward_match:
                    print(f"    Rewards differ: tau2={tau2_reward:.4f}, gym={gym_reward:.4f}")
                if not term_match:
                    print(f"    Terminated differs: tau2={tau2_term}, gym={gym_term}")
                if not trunc_match:
                    print(f"    Truncated differs: tau2={tau2_trunc}, gym={gym_trunc}")
                all_match = False

            # Stop if episode ended
            if tau2_term or tau2_trunc or gym_term or gym_trunc:
                print_info(f"  Episode ended at step {i}")
                break

        tau2_env.close()
        gym_env.close()

        return all_match

    except Exception as e:
        print_error(f"Step test failed: {e}")
        traceback.print_exc()
        return False


def test_train_test_splits(domain: str = "retail") -> bool:
    """Test that train/test splits are loaded correctly."""
    print_info(f"Testing train/test splits for domain: {domain}")

    try:
        # Test train split
        train_config = Tau2GymConfig(domain=domain, task_split="train")
        train_env = Tau2Env(train_config)
        train_task_count = len(train_env.task_ids)
        print_success(f"Train split loaded: {train_task_count} tasks")

        # Test test split
        test_config = Tau2GymConfig(domain=domain, task_split="test")
        test_env = Tau2Env(test_config)
        test_task_count = len(test_env.task_ids)
        print_success(f"Test split loaded: {test_task_count} tasks")

        # Test base split
        base_config = Tau2GymConfig(domain=domain, task_split="base")
        base_env = Tau2Env(base_config)
        base_task_count = len(base_env.task_ids)
        print_success(f"Base split loaded: {base_task_count} tasks")

        # Verify counts
        if train_task_count + test_task_count <= base_task_count:
            print_success("Split counts are consistent")
            counts_ok = True
        else:
            print_error(f"Split counts inconsistent: train({train_task_count}) + test({test_task_count}) > base({base_task_count})")
            counts_ok = False

        # Verify no overlap between train and test
        train_set = set(train_env.task_ids)
        test_set = set(test_env.task_ids)
        overlap = train_set & test_set

        if len(overlap) == 0:
            print_success("No overlap between train and test splits")
            no_overlap = True
        else:
            print_error(f"Found {len(overlap)} overlapping tasks between train and test")
            no_overlap = False

        train_env.close()
        test_env.close()
        base_env.close()

        return counts_ok and no_overlap

    except Exception as e:
        print_error(f"Split test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests(domains: List[str] = None) -> Tuple[int, int]:
    """Run all verification tests.

    Returns:
        Tuple of (passed_count, total_count)
    """
    if domains is None:
        domains = ["mock"]  # Start with mock domain for quick testing

    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"TAU2GYM VERIFICATION TEST SUITE")
    print(f"{'='*70}{Colors.ENDC}\n")

    results = []

    # Test 1: Environment Creation
    print(f"\n{Colors.BOLD}Test 1: Environment Creation{Colors.ENDC}")
    print("-" * 70)
    for domain in domains:
        result = test_environment_creation(domain)
        results.append(("Environment Creation", domain, result))

    # Test 2: Reset Identical
    print(f"\n{Colors.BOLD}Test 2: Reset Produces Identical Results{Colors.ENDC}")
    print("-" * 70)
    for domain in domains:
        result = test_reset_identical(domain)
        results.append(("Reset Identical", domain, result))

    # Test 3: Step Identical
    print(f"\n{Colors.BOLD}Test 3: Step Produces Identical Results{Colors.ENDC}")
    print("-" * 70)
    for domain in domains:
        result = test_step_identical(domain, num_steps=3)
        results.append(("Step Identical", domain, result))

    # Test 4: Train/Test Splits
    print(f"\n{Colors.BOLD}Test 4: Train/Test Splits{Colors.ENDC}")
    print("-" * 70)
    # Only test splits for domains that have them (skip mock)
    for domain in [d for d in domains if d != "mock"]:
        result = test_train_test_splits(domain)
        results.append(("Train/Test Splits", domain, result))

    # Print summary
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}{Colors.ENDC}\n")

    passed = sum(1 for _, _, result in results if result)
    total = len(results)

    for test_name, domain, result in results:
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if result else f"{Colors.RED}FAIL{Colors.ENDC}"
        print(f"{test_name:30} | {domain:10} | {status}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.ENDC}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.ENDC}")
        print(f"{Colors.GREEN}Tau2Gym produces identical results to tau2-bench.{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.ENDC}")
        print(f"{Colors.RED}Please review the errors above.{Colors.ENDC}\n")

    return passed, total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify Tau2Gym produces identical results to tau2-bench")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["mock"],
        choices=["mock", "retail", "airline", "telecom"],
        help="Domains to test (default: mock)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all domains"
    )

    args = parser.parse_args()

    if args.all:
        domains = ["mock", "retail", "airline", "telecom"]
    else:
        domains = args.domains

    passed, total = run_all_tests(domains)

    sys.exit(0 if passed == total else 1)
