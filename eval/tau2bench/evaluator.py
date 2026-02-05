"""
Tau2Bench evaluator that integrates with tau2bench framework
while supporting both VLLM-hosted and API models.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import tempfile
import shutil

from model_adapter import get_model_adapter, ModelAdapter


@dataclass
class Tau2EvalConfig:
    """Configuration for Tau2Bench evaluation."""

    # Model configuration
    model_name: str
    backend: str = "auto"  # "vllm", "openai", "litellm", "auto"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192

    # Evaluation configuration
    domains: List[str] = None  # ["retail", "airline", "telecom"]
    task_split: str = "test"  # "train", "test", "base"
    num_trials: int = 4  # Number of trials per task (for Pass^k)
    max_steps: int = 100  # Maximum conversation turns
    max_concurrency: int = 4  # Maximum concurrent evaluations

    # User simulator configuration
    user_model_name: str = "gpt-4o-mini"
    user_backend: str = "openai"
    user_base_url: Optional[str] = None  # e.g., ""
    user_api_key: Optional[str] = None  # API key for user simulator
    user_temperature: float = 1.0

    # Output configuration
    output_dir: str = "outputs/tau2bench"
    experiment_name: Optional[str] = None

    # Tau2bench configuration
    tau2_data_dir: Optional[str] = None  # Path to tau2-bench/data
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.domains is None:
            self.domains = ["retail", "airline", "telecom"]
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_{self.task_split}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class Tau2BenchEvaluator:
    """Evaluator for Tau2Bench using standardized model adapters."""

    def __init__(self, config: Tau2EvalConfig):
        self.config = config
        self.agent_adapter = self._create_agent_adapter()
        self.user_adapter = self._create_user_adapter()

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup tau2 data directory
        if config.tau2_data_dir:
            os.environ["TAU2_DATA_DIR"] = config.tau2_data_dir

        # Save configuration
        self._save_config()

    def _create_agent_adapter(self) -> ModelAdapter:
        """Create adapter for the agent model."""
        return get_model_adapter(
            model_name=self.config.model_name,
            backend=self.config.backend,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _create_user_adapter(self) -> ModelAdapter:
        """Create adapter for the user simulator model."""
        return get_model_adapter(
            model_name=self.config.user_model_name,
            backend=self.config.user_backend,
            base_url=self.config.user_base_url,
            api_key=self.config.user_api_key,
            temperature=self.config.user_temperature,
        )

    def _save_config(self):
        """Save evaluation configuration."""
        config_path = self.output_dir / "eval_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"✓ Configuration saved to {config_path}")

    def _get_litellm_model_string(self, adapter: ModelAdapter) -> str:
        """
        Convert adapter to LiteLLM model string for tau2 CLI.

        For tau2bench, we need to provide model strings in LiteLLM format:
        - OpenAI: "gpt-4o"
        - Custom: "openai/<model_name>" with custom base_url in args
        - VLLM: "openai/<model_name>" with custom base_url in args
        """
        model_info = adapter.get_model_info()
        model_name = model_info["model_name"]
        return model_name

    def _setup_environment_for_tau2(self):
        """
        Setup environment variables for tau2 CLI.

        IMPORTANT: For VLLM models, we need separate API configurations:
        - Agent uses VLLM URL (http://localhost:8500/v1) - passed via --agent-llm-args
        - User uses OpenAI-compatible API (e.g., ) - passed via --user-llm-args

        Since tau2's generate() function now prioritizes kwargs over environment variables,
        the agent and user configurations are properly separated via llm-args.
        Environment variables are set here as fallback/defaults only.
        """
        env = os.environ.copy()

        # Setup environment for USER SIMULATOR (as fallback/default)
        user_info = self.user_adapter.get_model_info()
        if user_info["backend"] == "openai":
            # User simulator uses OpenAI API
            if user_info.get("api_key"):
                env["OPENAI_API_KEY"] = user_info["api_key"]
            # base_url for user will be passed via --user-llm-args, which takes priority

        # Note: Agent's base_url and api_key are passed via --agent-llm-args,
        # which takes priority over environment variables, so setting them here
        # won't interfere with agent configuration.

        # Disable LiteLLM logging
        env["LITELLM_LOG"] = "ERROR"

        return env

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete Tau2Bench evaluation across all configured domains.

        Returns:
            Dictionary containing evaluation results and metrics
        """
        print("\n" + "="*80)
        print("Tau2Bench Evaluation")
        print("="*80)
        print(f"Agent Model: {self.config.model_name} ({self.config.backend})")
        print(f"User Model: {self.config.user_model_name}")
        print(f"Domains: {', '.join(self.config.domains)}")
        print(f"Task Split: {self.config.task_split}")
        print(f"Num Trials: {self.config.num_trials}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80 + "\n")

        # Setup environment
        env = self._setup_environment_for_tau2()

        # Get model strings for tau2 CLI
        agent_llm_string = self._get_litellm_model_string(self.agent_adapter)
        user_llm_string = self._get_litellm_model_string(self.user_adapter)

        results = {}
        trajectory_files = []

        # Run evaluation for each domain
        for domain in self.config.domains:
            print(f"\n{'='*80}")
            print(f"Evaluating domain: {domain}")
            print(f"{'='*80}\n")

            # Prepare output file name
            output_name = f"{self.config.experiment_name}_{domain}"
            output_file = self.output_dir / f"{output_name}.json"

            # Build tau2 run command
            cmd = [
                "tau2", "run",
                "--domain", domain,
                "--agent-llm", agent_llm_string,
                "--user-llm", user_llm_string,
                "--num-trials", str(self.config.num_trials),
                "--max-steps", str(self.config.max_steps),
                "--max-concurrency", str(self.config.max_concurrency),
                "--save-to", output_name,
            ]

            # Add task split (important: use test split for evaluation!)
            if self.config.task_split != "base":
                cmd.extend(["--task-split-name", self.config.task_split])

            # Add seed if specified
            if self.config.seed is not None:
                cmd.extend(["--seed", str(self.config.seed)])

            # Add llm-args for agent (temperature + base_url for VLLM + max_context_tokens)
            agent_info = self.agent_adapter.get_model_info()
            agent_args = {"temperature": self.config.temperature}
            
            # Add max_context_tokens if specified (default: 32768 for most models)
            # This is used by the truncation logic in tau2-bench's generate function
            max_context_tokens = getattr(self.config, "max_context_tokens", None)
            model_name = self.config.model_name.lower()
            if max_context_tokens is None:
                # Default based on model name
                if "gpt-4o" in model_name or "gpt-4-turbo" in model_name:
                    max_context_tokens = 128000
                else:
                    max_context_tokens = 32768
            agent_args["max_context_tokens"] = max_context_tokens

            # For qwen3 series models, disable thinking for non-streaming calls
            if "qwen3" in model_name:
                agent_args["enable_thinking"] = False

            if agent_info["backend"] == "vllm":
                # For VLLM, specify custom base URL and API key
                agent_args["base_url"] = agent_info["base_url"]
                agent_args["api_key"] = "EMPTY"
            elif agent_info["backend"] == "openai" and agent_info["base_url"] != "https://api.openai.com/v1":
                # For custom OpenAI-compatible endpoints
                agent_args["base_url"] = agent_info["base_url"]

            cmd.extend([
                "--agent-llm-args",
                json.dumps(agent_args)
            ])

            # Add llm-args for user (temperature + base_url if custom)
            user_info = self.user_adapter.get_model_info()
            user_args = {"temperature": self.config.user_temperature}

            if user_info["backend"] == "openai" and user_info["base_url"] != "https://api.openai.com/v1":
                # For custom OpenAI-compatible endpoints (e.g., "")
                user_args["base_url"] = user_info["base_url"]

            cmd.extend([
                "--user-llm-args",
                json.dumps(user_args)
            ])

            print(f"Running command: {' '.join(cmd)}\n")

            try:
                # Run tau2 evaluation with real-time output
                result = subprocess.run(
                    cmd,
                    env=env,
                    check=True,
                    cwd=os.getcwd()  # Run from current directory
                )

                # Check if output file was created
                # tau2 saves to data/tau2/simulations/ by default
                tau2_output = Path("data/tau2/simulations") / f"{output_name}.json"
                if tau2_output.exists():
                    # Copy to our output directory
                    shutil.copy(tau2_output, output_file)
                    trajectory_files.append(output_file)
                    print(f"✓ Results saved to {output_file}")

                    # Load and store results
                    with open(output_file) as f:
                        domain_results = json.load(f)
                    results[domain] = domain_results
                else:
                    print(f"⚠ Warning: Output file not found at {tau2_output}")
                    results[domain] = {"error": "Output file not found"}

            except subprocess.CalledProcessError as e:
                print(f"\n✗ Error evaluating {domain}: {e}")
                results[domain] = {"error": str(e)}

            print(f"\n{'='*80}\n")

        # Save summary results
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "config": self.config.to_dict(),
                "domains": list(results.keys()),
                "trajectory_files": [str(f) for f in trajectory_files],
            }, f, indent=2)

        print(f"\n✓ Evaluation complete!")
        print(f"✓ Summary saved to {summary_file}")
        print(f"✓ Trajectory files: {len(trajectory_files)}")

        return {
            "config": self.config,
            "results": results,
            "trajectory_files": trajectory_files,
            "summary_file": summary_file,
        }

    def compute_metrics(self, trajectory_files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Compute Pass^k metrics from trajectory files.

        Args:
            trajectory_files: List of trajectory file paths. If None, uses all files in output dir.

        Returns:
            Dictionary containing computed metrics
        """
        if trajectory_files is None:
            # Find all trajectory JSON files in output directory
            trajectory_files = list(self.output_dir.glob("*_[a-z]*.json"))

        if not trajectory_files:
            print("⚠ No trajectory files found for metric computation")
            return {}

        print(f"\nComputing metrics from {len(trajectory_files)} trajectory files...")

        # Use tau2's metric computation
        # We can load the JSON files and compute Pass^k ourselves or use tau2 submit commands

        metrics_by_domain = {}
        for traj_file in trajectory_files:
            with open(traj_file) as f:
                data = json.load(f)

            domain = data.get("info", {}).get("environment_info", {}).get("domain_name", "unknown")
            simulations = data.get("simulations", [])

            # Compute Pass^k metrics
            metrics = self._compute_passk_metrics(simulations, k_values=[1, 2, 3, 4])
            metrics_by_domain[domain] = metrics

        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_by_domain, f, indent=2)

        print(f"✓ Metrics saved to {metrics_file}")

        return metrics_by_domain

    def _compute_passk_metrics(self, simulations: List[Dict], k_values: List[int]) -> Dict[str, float]:
        """
        Compute Pass^k success rates.

        Pass^k: Probability of success in at least one of k trials
        Pass^k = # tasks with >= 1 success in k trials / # tasks
        """
        # Group simulations by task_id
        task_trials = {}
        for sim in simulations:
            task_id = sim.get("task_id")
            reward = sim.get("reward_info", {}).get("reward", 0.0)

            if task_id not in task_trials:
                task_trials[task_id] = []
            task_trials[task_id].append(reward)

        metrics = {}
        num_tasks = len(task_trials)

        for k in k_values:
            successes = 0
            for task_id, rewards in task_trials.items():
                # Check if any of the first k trials succeeded (reward >= 0.99)
                k_rewards = rewards[:k]
                if any(r >= 0.99 for r in k_rewards):
                    successes += 1

            pass_k_rate = successes / num_tasks if num_tasks > 0 else 0.0
            metrics[f"pass@{k}"] = pass_k_rate

        # Also compute average reward
        all_rewards = [r for rewards in task_trials.values() for r in rewards]
        metrics["average_reward"] = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        metrics["num_tasks"] = num_tasks
        metrics["num_simulations"] = len(simulations)

        return metrics
