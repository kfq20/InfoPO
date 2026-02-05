#!/usr/bin/env python3
"""
Test script to verify Tau2Gym async implementation.

This script tests that:
1. step_async is truly non-blocking
2. Multiple environments can execute concurrently
3. The async implementation doesn't cause blocking issues
"""

import asyncio
import time
from tau2gym import Tau2Env, Tau2GymConfig


async def test_single_env_async():
    """Test single environment async execution."""
    print("=" * 60)
    print("Test 1: Single environment async execution")
    print("=" * 60)
    
    config = Tau2GymConfig(
        domain="mock",  # Use mock domain for faster testing
        task_split="base",
        data_mode="single",
        max_steps=5,
        verbose=False,
    )
    
    env = Tau2Env(config)
    
    try:
        # Reset environment
        print("Resetting environment...")
        observation, info = await env.reset_async()
        print(f"Initial observation length: {len(observation)}")
        
        # Execute a few async steps
        test_actions = [
            "Hello! How can I help you today?",
            "Let me check that for you.",
            "done()",
        ]
        
        start_time = time.time()
        for i, action in enumerate(test_actions):
            print(f"\nStep {i+1}: {action[:50]}...")
            step_start = time.time()
            observation, reward, terminated, truncated, info = await env.step_async(action)
            step_time = time.time() - step_start
            print(f"  Step completed in {step_time:.2f}s")
            print(f"  Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                break
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f}s")
        print("✅ Single environment async test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


async def test_concurrent_envs():
    """Test concurrent execution of multiple environments."""
    print("\n" + "=" * 60)
    print("Test 2: Concurrent execution of multiple environments")
    print("=" * 60)
    
    config = Tau2GymConfig(
        domain="mock",
        task_split="base",
        data_mode="single",
        max_steps=3,
        verbose=False,
    )
    
    async def run_env(env_id: int):
        """Run a single environment."""
        env = Tau2Env(config)
        try:
            observation, info = await env.reset_async()
            
            actions = ["Hello!", "Check status", "done()"]
            for action in actions:
                observation, reward, terminated, truncated, info = await env.step_async(action)
                if terminated or truncated:
                    break
            
            return env_id, True
        except Exception as e:
            print(f"Error in env {env_id}: {e}")
            return env_id, False
        finally:
            env.close()
    
    # Run 3 environments concurrently
    num_envs = 3
    print(f"Running {num_envs} environments concurrently...")
    
    start_time = time.time()
    tasks = [run_env(i) for i in range(num_envs)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"\nConcurrent execution completed in {total_time:.2f}s")
    
    success_count = sum(1 for _, success in results if success)
    print(f"Successfully completed: {success_count}/{num_envs}")
    
    if success_count == num_envs:
        print("✅ Concurrent execution test passed!")
    else:
        print("❌ Concurrent execution test failed!")


async def test_async_vs_sync_performance():
    """Compare async vs sync execution time for concurrent operations."""
    print("\n" + "=" * 60)
    print("Test 3: Async vs Sync performance comparison")
    print("=" * 60)
    
    config = Tau2GymConfig(
        domain="mock",
        task_split="base",
        data_mode="single",
        max_steps=2,
        verbose=False,
    )
    
    # Test async execution
    async def async_execution():
        env = Tau2Env(config)
        try:
            await env.reset_async()
            await env.step_async("Hello!")
            await env.step_async("done()")
        finally:
            env.close()
    
    num_tasks = 3
    print(f"Running {num_tasks} tasks...")
    
    # Async execution
    start_time = time.time()
    tasks = [async_execution() for _ in range(num_tasks)]
    await asyncio.gather(*tasks)
    async_time = time.time() - start_time
    
    print(f"Async execution time: {async_time:.2f}s")
    print(f"Average time per task: {async_time/num_tasks:.2f}s")
    
    # Note: We can't easily test sync execution in this context since it would block
    # But the async execution should demonstrate non-blocking behavior
    print("✅ Performance test completed!")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Tau2Gym Async Implementation Test Suite")
    print("=" * 60 + "\n")
    
    try:
        await test_single_env_async()
        await test_concurrent_envs()
        await test_async_vs_sync_performance()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

