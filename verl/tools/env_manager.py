"""
Environment Manager for Multi-turn Interactions
Handles environment lifecycle and persistence across conversation turns.
"""

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment instances for multi-turn conversations."""
    
    def __init__(self):
        self._environments: Dict[str, Any] = {}  # request_id -> environment
        self._env_configs: Dict[str, Dict] = {}   # request_id -> config
    
    def create_environment(self, request_id: str, env_name: str, **kwargs) -> str:
        """Create and store environment for a conversation.
        
        Args:
            request_id: Unique identifier for the conversation
            env_name: Type of environment (TurtleGym, TelepathyGym, etc.)
            **kwargs: Environment-specific configuration
            
        Returns:
            request_id for the created environment
        """
        if request_id in self._environments:
            logger.warning(f"Environment for request_id {request_id} already exists")
            return request_id
            
        env = None
        
        if env_name == "TurtleGym":
            env = self._create_turtlegym_environment(**kwargs)
        elif env_name == "TelepathyGym":
            env = self._create_telepathygym_environment(**kwargs)
        elif env_name == "PersuadeGym":
            env = self._create_persuadegym_environment(**kwargs)
        elif env_name == "IntentionGym":
            env = self._create_intentiongym_environment(**kwargs)
        elif env_name == "TravelGym":
            env = self._create_travelgym_environment(**kwargs)
        elif env_name == "SearchGym":
            env = self._create_searchgym_environment(**kwargs)
        elif env_name == "TauGym":
            env = self._create_taugym_environment(**kwargs)
        elif env_name == "FunctionGym":
            env = self._create_functiongym_environment(**kwargs)
        elif env_name == "Tau2Gym":
            env, initial_observation = self._create_tau2gym_environment(**kwargs)
            self._environments[request_id] = env
            self._env_configs[request_id] = {
                "env_name": env_name,
                "kwargs": kwargs
            }
            logger.info(f"Created {env_name} environment for request {request_id}")
            return initial_observation  # Return initial observation for Tau2Gym
        elif env_name == "ColBenchCodeEnv":
            env = self._create_colbenchcodeenv_environment(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_name}")
        
        self._environments[request_id] = env
        self._env_configs[request_id] = {
            "env_name": env_name,
            "kwargs": kwargs
        }
        
        logger.info(f"Created {env_name} environment for request {request_id}")
        return request_id
    
    def get_environment(self, request_id: str) -> Optional[Any]:
        """Get environment for a conversation."""
        return self._environments.get(request_id)
    
    def release_environment(self, request_id: str) -> None:
        """Clean up environment for a conversation."""
        if request_id in self._environments:
            # Cleanup environment if it has cleanup methods
            env = self._environments[request_id]
            if hasattr(env, 'close'):
                env.close()
            
            del self._environments[request_id]
            del self._env_configs[request_id]
            logger.info(f"Released environment for request {request_id}")
    
    def _create_turtlegym_environment(self, **kwargs):
        """Create TurtleGym environment."""
        import turtlegym
        
        env_config = turtlegym.get_default_config()
        
        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        
        env_config.max_steps = max_turns
        env_config.success_threshold = 1.0
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name
        
        env = turtlegym.StoryEnv(config=env_config)
        env.reset()
        
        return env
    
    def _create_telepathygym_environment(self, **kwargs):
        """Create TelepathyGym environment."""
        import telepathygym
        
        env_config = telepathygym.get_default_config()
        
        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name

        env = telepathygym.TelepathyEnv(config=env_config)
        env.reset()
        
        return env

    def _create_persuadegym_environment(self, **kwargs):
        """Create PersuadeGym environment."""
        import persuadegym
        
        env_config = persuadegym.get_default_config()
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        
        env = persuadegym.PersuadeEnv(config=env_config)
        env.reset()
        
        return env

    def _create_intentiongym_environment(self, **kwargs):
        """Create IntentionGym environment."""
        import intentiongym
        
        env_config = intentiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name

        env = intentiongym.IntentionEnv(config=env_config)
        env.reset()

        return env

    def _create_travelgym_environment(self, **kwargs):
        """Create TravelGym environment."""
        import travelgym
        
        env_config = travelgym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        env_config.one_choice_per_aspect = True

        env_config.search_correct_reward = 0.0
        env_config.preference_correct_reward = 0.8

        env = travelgym.TravelEnv(config=env_config)
        env.reset()

        return env

    def _create_searchgym_environment(self, **kwargs):
        """Create SearchGym environment."""
        import searchgym
        
        env_config = searchgym.get_default_config()
        model_name = kwargs.get("model_name", "gpt-4o")
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.eval_method = "llm"
        env_config.model_name = model_name

        env = searchgym.SearchEnv(config=env_config)
        env.reset()

        return env
    
    def _create_taugym_environment(self, **kwargs):
        """Create TauGym environment."""
        import taugym
        
        env_config = taugym.get_default_config()
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.user_model = model_name

        if "retail" in id:
            env_config.task_category = "retail"
        else:
            env_config.task_category = "airline"
        if "train" in id:
            env_config.task_split = "train"
        else:
            env_config.task_split = "test"

        env = taugym.TauEnv(config=env_config)
        env.reset()
        
        return env
    
    def _create_functiongym_environment(self, **kwargs):
        """Create FunctionGym environment."""
        import functiongym
        
        env_config = functiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        
        env = functiongym.FunctionEnv(config=env_config)
        env.reset()

        return env

    def _create_tau2gym_environment(self, **kwargs):
        """Create Tau2Gym environment and return initial observation.

        Returns:
            Tuple of (env, initial_observation_str)
        """
        import tau2gym
        import os

        env_config = tau2gym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        domain = kwargs.get("domain", "retail")
        task_id = kwargs.get("id")
        solo_mode = kwargs.get("solo_mode", False)
        
        # Model name: prefer kwargs.model_name, then MULTITURN_MODEL_NAME env var, then default
        model_name = (kwargs.get("model_name") or 
                     os.getenv("MULTITURN_MODEL_NAME") or 
                     None)
        if model_name:
            env_config.user_llm = model_name
            logger.info(f"Tau2Gym user_llm set to: {model_name}")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = task_id
        env_config.domain = domain
        env_config.solo_mode = solo_mode

        logger.info(f"Tau2Gym environment config: max_steps={max_turns}, domain={domain}, task_id={task_id}, solo_mode={solo_mode}, user_llm={env_config.user_llm}")

        # Create environment
        env = tau2gym.Tau2Env(config=env_config)

        # Reset and get initial observation (user's first message from simulator)
        observation, info = env.reset()

        # Format observation to string if it's a dict
        if isinstance(observation, dict):
            initial_observation = observation.get("feedback", str(observation))
        else:
            initial_observation = str(observation)

        logger.info(f"Tau2Gym environment created with initial observation: {initial_observation[:100]}...")

        return env, initial_observation

    def _create_colbenchcodeenv_environment(self, **kwargs):
        """Create ColBenchCodeEnv environment."""
        import os
        import colbenchgym
        
        env_config = colbenchgym.get_code_config()
        
        # Configure from kwargs or environment variables
        max_turns = kwargs.get("max_turns", 10)
        
        # Model name: prefer MULTITURN_MODEL_NAME (unified), then COLBENCH_ENV_MODEL_NAME (backward compat)
        model_name = (kwargs.get("model_name") or 
                     os.getenv("MULTITURN_MODEL_NAME") or 
                     os.getenv("COLBENCH_ENV_MODEL_NAME", ""))
        
        # Base URL: prefer OPENAI_BASE_URL (unified), then OPENAI_API_BASE (alias), then COLBENCH_ENV_BASE_URL (backward compat)
        base_url = (kwargs.get("base_url") or 
                   os.getenv("OPENAI_BASE_URL") or 
                   os.getenv("OPENAI_API_BASE") or  # Support both OPENAI_BASE_URL and OPENAI_API_BASE
                   os.getenv("COLBENCH_ENV_BASE_URL", ""))
        
        if base_url:
            # Use provided base_url directly
            env_config.env_base_url = base_url
            # Extract hostname and port for backward compatibility (optional)
            if "://" in base_url:
                # Parse URL to extract hostname and port
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                env_config.env_hostname = parsed.hostname or "localhost"
                env_config.env_port = parsed.port or 8000
        else:
            # Fallback to hostname:port format
            hostname = kwargs.get("hostname") or os.getenv("COLBENCH_ENV_HOSTNAME", "localhost")
            port = kwargs.get("port") or int(os.getenv("COLBENCH_ENV_PORT", "8000"))
            env_config.env_hostname = hostname
            env_config.env_port = port
        
        # API key: prefer OPENAI_API_KEY (unified), then COLBENCH_ENV_API_KEY (backward compat)
        api_key = (kwargs.get("api_key") or 
                  os.getenv("OPENAI_API_KEY") or 
                  os.getenv("COLBENCH_ENV_API_KEY", ""))
        env_config.env_api_key = api_key
        
        # Get task information
        # Task can be passed directly as a dict, or as individual fields
        task = kwargs.get("task", {})
        if isinstance(task, dict):
            problem_description = task.get("problem_description", kwargs.get("problem_description", ""))
            ground_truth = task.get("ground_truth", kwargs.get("ground_truth", ""))
            test_cases = task.get("test_cases", kwargs.get("test_cases", {}))
        else:
            problem_description = kwargs.get("problem_description", "")
            ground_truth = kwargs.get("ground_truth", "")
            test_cases = kwargs.get("test_cases", {})
        
        task_id = kwargs.get("id")
        category = kwargs.get("category")
        
        # Configure environment
        env_config.max_steps = max_turns
        env_config.env_model_name = model_name
        env_config.verbose = kwargs.get("verbose", False)
        
        # Create task dict for environment (include test_cases if available)
        task = {
            "problem_description": problem_description,
            "ground_truth": ground_truth,
            "test_cases": test_cases
        }
        
        # Create environment
        env = colbenchgym.ColBenchCodeEnv(
            config=env_config,
            task=task,
            category=category,
            id=task_id
        )
        
        # Reset environment with task
        env.reset(options={"task": task})
        
        return env


# Global environment manager instance
_env_manager = EnvironmentManager()


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    return _env_manager 