#!/usr/bin/env python3
"""
ColBench evaluation script for OpenAI-compatible API models.
Both agent and user simulator use API models (no VLLM server needed).
"""
import os
import sys
import json
import concurrent.futures
from pathlib import Path
from fire import Fire
from tqdm import tqdm
from openai import OpenAI

# Add sweet_rl to path
SWEET_RL_DIR = Path(__file__).parent.parent.parent.parent / "sweet_rl"
sys.path.insert(0, str(SWEET_RL_DIR))


def check_and_extract_answer(response: str):
    """
    Check if response contains answer marker and extract the answer.
    Supports multiple formats:
    - "I WANT TO ANSWER:" (standard format)
    - "I WANT_TO_ANSWER:" (underscore format)
    - "I WANT_TO ANSWER:" (mixed format)
    - "i want to answer:" (lowercase)
    - etc.
    
    Returns:
        tuple: (has_answer: bool, answer_text: str)
        - has_answer: True if answer marker found
        - answer_text: Extracted answer text (empty string if not found)
    """
    if not response:
        return False, ""
    
    # Normalize the response for pattern matching (case-insensitive)
    response_lower = response.lower()
    
    # Try to find answer marker in various formats
    patterns = [
        "I WANT TO ANSWER:",  # Standard format
        "I WANT_TO_ANSWER:",  # Underscore format
        "I WANT_TO ANSWER:",   # Mixed format
        "i want to answer:",   # Lowercase
        "i want_to_answer:",   # Lowercase with underscore
    ]
    
    for pattern in patterns:
        # Case-insensitive search
        pattern_lower = pattern.lower()
        if pattern_lower in response_lower:
            # Find the actual position in original response (case-sensitive)
            # Try exact match first
            if pattern in response:
                idx = response.find(pattern)
                answer_text = response[idx + len(pattern):].strip()
                return True, answer_text
            # If exact match fails, try case-insensitive search
            else:
                # Find position in lowercase version
                idx_lower = response_lower.find(pattern_lower)
                # Extract from original response at same position
                answer_text = response[idx_lower + len(pattern):].strip()
                return True, answer_text
    
    return False, ""


class OpenAIAgent:
    """Simple wrapper for OpenAI-compatible API models."""

    def __init__(self, model_name: str, agent_prompt: str, api_key: str, base_url: str, temperature: float = 0.0):
        self.model_name = model_name
        self.agent_prompt = agent_prompt
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_action(self, batch_obs):
        """Get actions for a batch of observations (parallel API calls)."""
        formatted_prompts = []
        responses = []

        print(f"[AGENT] Processing {len(batch_obs)} observations in parallel...")

        def call_api_for_obs(idx, obs):
            """Call API for a single observation."""
            # Skip if observation is None (environment is done)
            if obs is None:
                return (idx, None, None, "Environment already done")
            
            # Format messages for OpenAI API
            # Training format: messages = [system_message, user_message, assistant_message, ...]
            # System message is fixed (no {dialogue_history} placeholder), dialogue history grows by appending messages
            # obs is a list of messages from get_dialogue_history() (already contains user/assistant messages)
            if isinstance(obs, list):
                # Build messages list matching training format:
                # 1. System message (from agent_prompt, remove {dialogue_history} placeholder if present)
                #    Note: Training data doesn't have this placeholder, but the prompt file does
                # 2. Dialogue history messages (user, assistant, user, ...)
                system_content = self.agent_prompt.replace("{dialogue_history}", "").strip()
                messages = [
                    {"role": "system", "content": system_content}
                ] + obs
            else:
                # Fallback: if obs is a string, treat as initial user message
                system_content = self.agent_prompt.replace("{dialogue_history}", "").strip()
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": str(obs)}
                ]

            # Call OpenAI API
            try:
                # For qwen3 models, disable thinking for non-streaming calls
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "timeout": 60.0,
                }
                if "qwen3" in self.model_name.lower():
                    params["extra_body"] = {"enable_thinking": False}
                
                response = self.client.chat.completions.create(**params)
                response_text = response.choices[0].message.content
                # Format prompt for logging (convert messages to string)
                prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                return (idx, prompt_str, response_text, None)
            except Exception as e:
                prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) if isinstance(obs, list) else str(obs)
                return (idx, prompt_str, "I WANT TO ANSWER: # Error occurred during API call", str(e))

        # Parallel API calls using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_obs), 10)) as executor:
            futures = [executor.submit(call_api_for_obs, idx, obs) for idx, obs in enumerate(batch_obs)]

            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                idx, prompt, response_text, error = future.result()
                results.append((idx, prompt, response_text, error))

                # Progress indicator
                completed = len(results)
                if error:
                    if error == "Environment already done":
                        print(f"[AGENT] {completed}/{len(batch_obs)} - Obs {idx+1} SKIPPED (done)")
                    else:
                        print(f"[AGENT] {completed}/{len(batch_obs)} - Obs {idx+1} FAILED: {error}")
                else:
                    print(f"[AGENT] {completed}/{len(batch_obs)} - Obs {idx+1} OK ({len(response_text)} chars)")

        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])

        # Extract prompts and responses in original order
        for idx, prompt, response_text, error in results:
            if prompt is None:
                # Environment is done, return placeholder response
                formatted_prompts.append("")
                responses.append("")
            else:
                formatted_prompts.append(prompt)
                responses.append(response_text)

        print(f"[AGENT] All {len(batch_obs)} observations processed")
        return formatted_prompts, responses


class APIHumanSimulator:
    """Human simulator using OpenAI-compatible API (replaces VLLM server)."""

    def __init__(self, model_name: str, human_prompt: str, api_key: str, base_url: str, env_id: int = 0, max_steps: int = 10):
        self.model_name = model_name
        self.human_prompt = human_prompt
        self.env_id = env_id
        self.max_steps = max_steps
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.problem_description = ""
        self.hidden_information = ""
        self.answer = "No answer"
        self.steps = 0
        self.done = False
        self.dialogue_history = []

    def get_dialogue_history(self):
        """Get dialogue history in message format."""
        messages = [
            {"role": d["role"], "content": d["content"]}
            for d in self.dialogue_history
        ]
        return messages

    def str_dialogue_history(self):
        """Format dialogue history as string."""
        result = ""
        for d in self.dialogue_history:
            result += str(d["role"]) + ":"
            result += str(d["content"]) + "\n\n\n\n"
        return result + "agent:"

    def reset(self, problem_description, hidden_information):
        """Reset the environment."""
        self.problem_description = str(problem_description)
        self.hidden_information = str(hidden_information)
        self.answer = "No answer"
        self.steps = 0
        self.done = False
        self.dialogue_history = []
        self.dialogue_history.append({
            "role": "user",
            "content": problem_description,
        })
        return self.get_dialogue_history()

    def invoke_model(self):
        """Call API to simulate human response."""
        HUMAN_RESPONSE_CHARACTER_LIMIT = 400

        for attempt in range(3):
            try:
                print(f"[USER SIM] Calling API (attempt {attempt+1}/3)...")
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": self.human_prompt.format(
                            problem_description=self.problem_description,
                            hidden_information=self.hidden_information,
                            dialogue_history=self.str_dialogue_history(),
                        ),
                    },
                ]
                # For qwen3 models, disable thinking for non-streaming calls
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "timeout": 60.0,
                }
                if "qwen3" in self.model_name.lower():
                    params["extra_body"] = {"enable_thinking": False}
                
                completion = self.client.chat.completions.create(**params)
                response = completion.choices[0].message.content[:HUMAN_RESPONSE_CHARACTER_LIMIT]
                print(f"[USER SIM] Got response ({len(response)} chars)")
                return response
            except Exception as e:
                print(f"[USER SIM] API error (attempt {attempt+1}): {e}")
                if attempt == 2:  # Last attempt
                    print(f"[USER SIM] All attempts failed, returning default response")
                continue
        return "No response."

    def step(self, response, formatted_prompt=None):
        """Step the environment with agent's response."""
        self.steps += 1
        if self.done:
            return None, 0, True

        raw_response = response

        if "OUTPUT:" in response:
            response = response.split("OUTPUT:")[1]
            raw_response = "OUTPUT:".join(raw_response.split("OUTPUT:")[:2])

        # Check for answer marker using flexible format detection
        has_answer, answer_text = check_and_extract_answer(response)
        
        if has_answer or self.steps >= self.max_steps:
            self.done = True
            if has_answer:
                self.answer = answer_text
            else:
                self.answer = response

        self.dialogue_history.append({
            "role": "assistant",
            "content": response,
            "input": formatted_prompt,
            "output": raw_response,
        })

        if not self.done:
            answer = self.invoke_model()
            self.dialogue_history.append(
                {"role": "user", "content": answer}
            )
        return self.get_dialogue_history() if not self.done else None, 0, self.done


def extract_answer_from_env(env):
    """Extract answer from environment, matching training-time fallback logic.
    
    This function replicates the answer extraction logic from ColBenchCodeEnv.calculate_reward()
    to ensure consistency between training and evaluation.
    """
    # If answer is already set and not "No answer", use it
    if env.answer != "No answer" and env.answer:
        return env.answer.strip()
    
    # Try to extract from dialogue history (fallback logic from training)
    if env.dialogue_history:
        # Find the last agent response
        for msg in reversed(env.dialogue_history):
            if msg.get("role") == "assistant":
                last_response = msg.get("content", "").strip()
                if last_response:
                    # Try to extract answer using flexible format detection
                    has_answer, answer_text = check_and_extract_answer(last_response)
                    if has_answer and answer_text:
                        return answer_text
                    # If no "I WANT TO ANSWER:" but we have a response that looks like code, use it as answer
                    # This handles cases where rollout ended before agent could format answer properly
                    # Check if response contains code-like patterns (def, import, class, etc.)
                    elif len(last_response) > 20 and any(keyword in last_response for keyword in ["def ", "import ", "class ", "return ", "="]):
                        return last_response
                    # If episode is done (reached max_steps) but no code pattern found, still use the last response as answer
                    # This handles cases where rollout ended early and agent didn't provide final answer
                    elif env.done and len(last_response) > 10:
                        return last_response
    
    # Fallback: return the stored answer even if it's "No answer"
    return env.answer.strip() if env.answer else "No answer"


def batch_interact_environment(agent, environments, tasks):
    """Run agent-environment interaction (same as sweet_rl)."""
    print(f"[DEBUG] Starting batch with {len(tasks)} tasks...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        print(f"[DEBUG] Resetting environments...")
        jobs = [
            executor.submit(
                env.reset, task["problem_description"], task["ground_truth"]
            )
            for env, task in zip(environments, tasks)
        ]
        batch_obs = [job.result() for job in jobs]
        print(f"[DEBUG] Environments reset. Starting interaction...")

    for j in range(environments[0].max_steps + 1):
        print(f"[DEBUG] Step {j+1}/{environments[0].max_steps + 1}")

        # Filter out None observations (done environments)
        active_indices = [i for i, obs in enumerate(batch_obs) if obs is not None]
        if not active_indices:
            print(f"[DEBUG]   All environments are done, breaking early")
            break

        print(f"[DEBUG]   Active environments: {len(active_indices)}/{len(batch_obs)}")
        
        # Get agent actions only for active environments
        active_obs = [batch_obs[i] for i in active_indices]
        formatted_prompts, responses = agent.get_action(active_obs)
        print(f"[DEBUG]   Got {len(responses)} agent responses")

        # Step environments
        print(f"[DEBUG]   Stepping environments...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = []
            response_idx = 0
            for i, env in enumerate(environments):
                if i in active_indices:
                    jobs.append(executor.submit(env.step, responses[response_idx], formatted_prompts[response_idx]))
                    response_idx += 1
                else:
                    # Environment is done, return None
                    jobs.append(executor.submit(lambda: (None, 0, True)))
            
            step_results = [job.result() for job in jobs]
            batch_obs = [result[0] for result in step_results]
        print(f"[DEBUG]   Environments stepped")

    print(f"[DEBUG] Interaction complete. Collecting results...")
    # Use extract_answer_from_env to ensure consistency with training-time logic
    return [
        {"task": task, "dialogue_history": env.dialogue_history, "answer": extract_answer_from_env(env)}
        for task, env in zip(tasks, environments)
    ]


def run_api_evaluation(
    agent_model: str = "gpt-4o",
    agent_api_key: str = None,
    agent_base_url: str = "https://api.openai.com/v1",
    agent_temperature: float = 0.0,
    agent_vllm_host: str = None,  # If set, agent uses VLLM server (host:port format, e.g., "localhost:8000")
    user_simulator_model: str = "gpt-4o-mini",
    user_simulator_api_key: str = None,
    user_simulator_base_url: str = "https://api.openai.com/v1",
    output_dir: str = "outputs/colbench",
    experiment_name: str = "api_test",
    num_tasks: int = 100,
    batch_size: int = 32,
    max_steps: int = 10,
    best_of_n: int = 1,
    task_type: str = "code",
    seed: int = None,  # Random seed for task sampling (None = use first num_tasks deterministically)
):
    """
    Run ColBench evaluation with API models or VLLM server for agent.

    Args:
        agent_model: Agent model name (API model name or VLLM model name)
        agent_api_key: Agent API key (or set OPENAI_API_KEY env var) - not needed for VLLM
        agent_base_url: Agent API base URL - not used if agent_vllm_host is set
        agent_temperature: Agent sampling temperature
        agent_vllm_host: If set, agent uses VLLM server (host:port, e.g., "localhost:8000")
        user_simulator_model: User simulator model name
        user_simulator_api_key: User simulator API key (defaults to agent_api_key)
        user_simulator_base_url: User simulator API base URL
        output_dir: Output directory
        experiment_name: Experiment name
        num_tasks: Number of tasks to evaluate
        batch_size: Batch size for parallel execution
        max_steps: Max conversation turns
        best_of_n: Number of samples per task
        task_type: Task type (code or html)
        seed: Random seed for task sampling. If None, uses first num_tasks deterministically.
    """
    # Configure agent: VLLM server or API
    if agent_vllm_host:
        # Agent uses VLLM server
        agent_base_url = f"http://{agent_vllm_host}/v1"
        agent_api_key = "EMPTY"  # VLLM doesn't require authentication
    else:
        # Agent uses API
        agent_api_key = agent_api_key or os.getenv("OPENAI_API_KEY")
        if not agent_api_key:
            raise ValueError("Agent API key not set (OPENAI_API_KEY or agent_api_key)")

    user_simulator_api_key = user_simulator_api_key or os.getenv("OPENAI_API_KEY")
    if not user_simulator_api_key:
        raise ValueError("User simulator API key not set (OPENAI_API_KEY or user_simulator_api_key)")

    # Setup paths
    output_dir = Path(output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectories.jsonl"

    # Load data
    import pandas as pd
    import random
    data_dir = Path(__file__).parent.parent.parent / "data" / "colbench_code"
    df = pd.read_parquet(data_dir / "test.parquet")

    # Convert to task format expected by environment
    # Sample tasks: if seed is provided, shuffle with seed for reproducibility
    # Otherwise, just take first num_tasks (deterministic, always same tasks)
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"[INFO] Using random seed {seed} for task sampling")
    else:
        print(f"[INFO] Using first {num_tasks} tasks (deterministic, no shuffle)")
    raw_tasks = df.to_dict('records')[:num_tasks]
    tasks = []
    for raw_task in raw_tasks:
        # Extract task info from reward_model field
        reward_model = raw_task['reward_model']

        # Get test cases from tools_kwargs if available
        test_cases = {}
        if 'extra_info' in raw_task and 'tools_kwargs' in raw_task['extra_info']:
            tools_kwargs = raw_task['extra_info']['tools_kwargs']
            if 'interact_with_env' in tools_kwargs:
                create_kwargs = tools_kwargs['interact_with_env'].get('create_kwargs', {})
                if 'task' in create_kwargs:
                    test_cases = create_kwargs['task'].get('test_cases', {})

        # Filter None values from test_cases (parquet serialization may add None values for schema consistency)
        valid_test_cases = {k: v for k, v in test_cases.items() if v is not None}

        # Create task in sweet_rl format
        task = {
            'problem_description': reward_model['problem_description'],
            'ground_truth': reward_model['ground_truth'],
            'test_cases': valid_test_cases,  # Use filtered test_cases
        }
        tasks.append(task)

    # Repeat for best_of_n
    tasks = tasks * best_of_n

    # Get prompts
    prompt_dir = SWEET_RL_DIR / "prompts"
    if task_type == "code":
        user_prompt_path = prompt_dir / "human_simulator_code_prompt.txt"
        agent_prompt_path = prompt_dir / "llm_agent_code_prompt.txt"
    else:
        user_prompt_path = prompt_dir / "human_simulator_html_prompt.txt"
        agent_prompt_path = prompt_dir / "llm_agent_html_prompt.txt"

    with open(user_prompt_path) as f:
        human_prompt = f.read()
    with open(agent_prompt_path) as f:
        agent_prompt = f.read()

    # Create API-based environments (no VLLM server needed)
    print("Creating API-based environments...")
    environments = [
        APIHumanSimulator(
            model_name=user_simulator_model,
            human_prompt=human_prompt,
            api_key=user_simulator_api_key,
            base_url=user_simulator_base_url,
            env_id=i,
            max_steps=max_steps,
        )
        for i in range(min(len(tasks), batch_size))
    ]

    # Create agent
    agent = OpenAIAgent(
        model_name=agent_model,
        agent_prompt=agent_prompt,
        api_key=agent_api_key,
        base_url=agent_base_url,
        temperature=agent_temperature,
    )

    print("=" * 80)
    if agent_vllm_host:
        print("ColBench Evaluation - VLLM Agent + API User Simulator")
    else:
        print("ColBench Evaluation - API Models")
    print("=" * 80)
    print(f"Agent Model: {agent_model}")
    if agent_vllm_host:
        print(f"Agent VLLM Server: {agent_vllm_host}")
    else:
        print(f"Agent API: {agent_base_url}")
    print(f"Agent Temperature: {agent_temperature}")
    print()
    print(f"User Simulator Model: {user_simulator_model}")
    print(f"User Simulator API: {user_simulator_base_url}")
    print()
    print(f"Num Tasks: {num_tasks}")
    print(f"Best-of-{best_of_n}")
    print(f"Output: {output_path}")
    print("=" * 80)
    print()

    # Run evaluation
    trajectory = []
    for i in tqdm(range(0, len(tasks), batch_size)):
        current_tasks = tasks[i : i + batch_size]
        trajectory.extend(
            batch_interact_environment(
                agent, environments[: len(current_tasks)], current_tasks
            )
        )

        # Save incrementally
        with open(output_path, "w") as f:
            for d in trajectory:
                f.write(json.dumps(d) + "\n")

    print()
    print("=" * 80)
    print("Trajectory generation complete!")
    print(f"Saved to: {output_path}")
    print()
    print("Next step: Run evaluation")
    print(f"  python run_evaluate.py --saved_path {output_path} --k {best_of_n}")
    print("=" * 80)


if __name__ == "__main__":
    Fire(run_api_evaluation)
