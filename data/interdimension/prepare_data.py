"""
Prepare training and validation datasets for the InterDimension environment.

Each record follows the structure expected by verl's RLHFDataset:
- `prompt`: list of chat messages (system/user pairs)
- `data_source`: identifier string
- `reward_model`: metadata for reward computation
- `extra_info`: miscellaneous info including tools_kwargs and indices
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

# Paths to environment instructions
ROOT_DIR = Path(__file__).resolve().parents[2]
GYM_DIR = ROOT_DIR / "gyms" / "InterDimensionGym" / "interdimensiongym"
INSTRUCTION_PATH = GYM_DIR / "config" / "agent_instruction.txt"
ACTION_SPACE_PATH = GYM_DIR / "config" / "action_space.txt"

INSTRUCTION_TEXT = INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
ACTION_SPACE_TEXT = ACTION_SPACE_PATH.read_text(encoding="utf-8").strip()

MAX_TURNS = 40
DATA_SOURCE = "interdimension"
TRAIN_LEVELS = list(range(1, 9))
VAL_LEVELS = list(range(9, 11))
TRAIN_DUPLICATION = 40   # Produces 8 * 40 = 320 training entries
VAL_DUPLICATION = 10     # 2 * 10 = 20 validation entries


def build_prompt(level_id: str) -> List[Dict[str, str]]:
    """Construct the chat prompt for a given level."""
    system_message = (
        f"{INSTRUCTION_TEXT}\n\n"
        "You must strictly follow the action specifications below and respond with JSON only.\n"
        "Action specifications:\n"
        f"{ACTION_SPACE_TEXT}\n\n"
        "Important rules:\n"
        "1. Always output exactly one JSON object with keys `action` and `params`.\n"
        "2. Do not include any additional commentary.\n"
        "3. Ensure all parameters are valid according to the action schema."
    )
    user_message = (
        "You will now play the InterDimensional Market environment. "
        f"Level ID: {level_id}.\n"
        "Wait for the environment observation each turn and reply with exactly one JSON action."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def create_entry(level_num: int, split: str, unique_index: int) -> Dict:
    level_id = f"level_{level_num:02d}"
    return {
        "data_source": DATA_SOURCE,
        "prompt": build_prompt(level_id),
        "reward_model": {"type": "env_reward"},
        "extra_info": {
            "environment": "InterDimensionalMarket",
            "index": unique_index,
            "level_id": level_id,
            "split": split,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "interact_with_env": {
                    "create_kwargs": {
                        "level_id": level_id,
                        "max_turns": MAX_TURNS,
                    }
                }
            },
        },
        "level_id": level_id,
        "level_number": level_num,
    }


def prepare_interdimension_data():
    """Prepare training and validation data with duplication for larger batch sizes."""
    train_entries: List[Dict] = []
    val_entries: List[Dict] = []

    unique_counter = 1
    for level in TRAIN_LEVELS:
        for _ in range(TRAIN_DUPLICATION):
            train_entries.append(create_entry(level, "train", unique_counter))
            unique_counter += 1

    for level in VAL_LEVELS:
        for _ in range(VAL_DUPLICATION):
            val_entries.append(create_entry(level, "val", unique_counter))
            unique_counter += 1

    train_df = pd.DataFrame(train_entries)
    val_df = pd.DataFrame(val_entries)

    output_dir = Path(__file__).parent
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Created training data: {len(train_df)} samples")
    print(f"Created validation data: {len(val_df)} samples")
    print(f"Files saved to {output_dir}")

    return train_df, val_df


if __name__ == "__main__":
    prepare_interdimension_data()

