# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import logging
import os
import json
from datetime import datetime
from uuid import uuid4
import numpy as np

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format.
    Handles numpy arrays, torch tensors, and nested structures.
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes
        return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
    else:
        # Fallback: convert to string
        return str(obj)


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False, gamma=0.9):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_diction = data_item.non_tensor_batch.get("reward_model", {}) or {}
            ground_truth = reward_diction.get("ground_truth")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "")
            if hasattr(data_source, "__len__") and not isinstance(data_source, str):
                data_source = data_source[0] if len(data_source) > 0 else ""
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # customized for interaction gym reward computation (already calculated in the rollout)
            # Note that the score here is only for logging purpose, not for training purpose
            # This handles InteractComp (interact_interactcomp) and ColBench (interact_colbench_code)
            if data_source.startswith("interact"):
                # For InteractComp and ColBench, we use the final trajectory reward from reward_scores
                # conversation_histories is available but only contains action/observation, not per-turn rewards
                reward_scores = data_item.non_tensor_batch.get("reward_scores", [{}])
                if isinstance(reward_scores, np.ndarray):
                    reward_scores = reward_scores[0]
                elif isinstance(reward_scores, list):
                    reward_scores = reward_scores[0] if reward_scores else {}

                # Extract the interact_with_env reward (should be 0.0 or 1.0)
                final_reward = reward_scores.get("interact_with_env", 0.0) if isinstance(reward_scores, dict) else 0.0

                score = {
                    "score": final_reward,
                    "score_max": final_reward,
                }

                # Extract conversation_histories for logging (if available)
                conversation_histories = []
                if "conversation_histories" in data_item.non_tensor_batch:
                    conv_hist_data = data_item.non_tensor_batch["conversation_histories"]
                    if isinstance(conv_hist_data, np.ndarray):
                        conversation_histories = conv_hist_data[0] if len(conv_hist_data) > 0 else []
                    elif isinstance(conv_hist_data, list):
                        conversation_histories = conv_hist_data[0] if conv_hist_data else []
                    else:
                        conversation_histories = conv_hist_data
            elif data_source == "interdimension" or reward_diction.get("type") == "env_reward":
                reward_scores = data_item.non_tensor_batch.get("reward_scores", [{}])
                if isinstance(reward_scores, np.ndarray):
                    reward_scores = reward_scores[0]
                elif isinstance(reward_scores, list):
                    reward_scores = reward_scores[0] if reward_scores else {}

                final_reward = 0.0
                if isinstance(reward_scores, dict):
                    final_reward = float(reward_scores.get("interact_with_env", final_reward))
                else:
                    try:
                        final_reward = float(reward_scores)
                    except (TypeError, ValueError):
                        logging.getLogger(__name__).warning(
                            "Unexpected reward_scores format for interdimension: %s", type(reward_scores)
                        )

                score = {
                    "score": final_reward,
                    "score_max": final_reward,
                }
                conversation_histories = []
                if "conversation_histories" in data_item.non_tensor_batch:
                    conv_hist_data = data_item.non_tensor_batch["conversation_histories"]
                    if isinstance(conv_hist_data, np.ndarray):
                        conversation_histories = conv_hist_data[0] if len(conv_hist_data) > 0 else []
                    elif isinstance(conv_hist_data, list):
                        conversation_histories = conv_hist_data[0] if conv_hist_data else []
                    else:
                        conversation_histories = conv_hist_data
                ground_truth = ground_truth if ground_truth is not None else ""
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=reward_diction,
                    extra_info=extra_info,
                )

            # Note that the score here is only for logging purpose, not for training purpose
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # Optional: dump evaluation trajectories for debugging when enabled
            dump_dir = os.environ.get("USERRL_EVAL_DUMP_DIR", None)
            if dump_dir and os.access(os.path.dirname(dump_dir) or ".", os.W_OK):
                try:
                    os.makedirs(dump_dir, exist_ok=True)

                    # Extract messages for detailed prompt inspection
                    messages = None
                    if "messages" in data_item.non_tensor_batch:
                        messages_data = data_item.non_tensor_batch["messages"]
                        if isinstance(messages_data, dict) and "messages" in messages_data:
                            messages = messages_data["messages"]
                        elif hasattr(messages_data, "item") and callable(messages_data.item):
                            msg_obj = messages_data.item()
                            if isinstance(msg_obj, dict) and "messages" in msg_obj:
                                messages = msg_obj["messages"]

                    record = {
                        "id": str(uuid4()),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "data_source": data_source,
                        "ground_truth": str(ground_truth) if not isinstance(ground_truth, (str, int, float, bool, type(None))) else ground_truth,
                        "prompt": prompt_str,
                        "response": response_str,
                        "reward": float(reward),
                        "prompt_length": len(valid_prompt_ids),
                        "response_length": len(valid_response_ids),
                    }
                    if data_source.startswith("interact"):
                        # Make conversation_histories JSON serializable (covers InteractComp and ColBench)
                        record["conversation_histories"] = make_json_serializable(conversation_histories)
                        record["num_turns"] = len(conversation_histories)
                    elif data_source == "interdimension":
                        record["conversation_histories"] = make_json_serializable(conversation_histories)
                        record["num_turns"] = len(conversation_histories)
                    if isinstance(score, dict):
                        record["score_detail"] = make_json_serializable(score)
                    if messages:
                        # Save messages for detailed prompt inspection
                        record["messages"] = [
                            {"role": msg.role if hasattr(msg, "role") else msg.get("role", "unknown"),
                             "content": str(msg.content if hasattr(msg, "content") else msg.get("content", ""))}
                            for msg in messages
                        ]

                    # Save structured JSONL for machine processing
                    with open(os.path.join(dump_dir, "eval_traj.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Save human-readable format for easy inspection
                    with open(os.path.join(dump_dir, "readable_logs.txt"), "a", encoding="utf-8") as f:
                        f.write("\n" + "="*100 + "\n")
                        f.write(f"ID: {record['id']} | Timestamp: {record['timestamp']}\n")
                        f.write(f"Data Source: {data_source} | Reward: {reward}\n")
                        f.write("="*100 + "\n")

                        # Write messages if available (shows each turn's prompt)
                        if messages:
                            f.write("\n[MESSAGES HISTORY]\n")
                            f.write("-"*100 + "\n")
                            for idx, msg in enumerate(messages):
                                msg_role = msg.role if hasattr(msg, "role") else msg.get("role", "unknown")
                                msg_content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                                f.write(f"\n[{idx}] Role: {msg_role}\n")
                                f.write(f"{msg_content}\n")
                                f.write("-"*100 + "\n")

                        f.write(f"\n[FULL PROMPT]\n{prompt_str}\n")
                        f.write(f"\n[FULL RESPONSE]\n{response_str}\n")
                        f.write(f"\n[GROUND TRUTH]\n{ground_truth}\n")

                        if data_source.startswith("interact") and len(conversation_histories) > 0:
                            f.write(f"\n[CONVERSATION HISTORIES - {len(conversation_histories)} turns] (InteractComp/ColBench)\n")
                        elif data_source == "interdimension" and len(conversation_histories) > 0:
                            f.write(f"\n[INTERDIMENSION HISTORIES - {len(conversation_histories)} turns]\n")

                        if isinstance(score, dict):
                            f.write(f"\n[SCORE DETAILS]\n")
                            for key, value in score.items():
                                f.write(f"  {key}: {value}\n")

                        f.write("\n" + "="*100 + "\n\n")

                except Exception as e:
                    print(f"[WARNING] Failed to dump evaluation trajectory: {e}")
                    import traceback
                    traceback.print_exc()

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
