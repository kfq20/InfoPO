#!/bin/bash
# Training script for Tau2Gym with UserRL
# This script demonstrates how to train an RL agent on tau2-bench tasks using multi-turn interaction

set -e  # Exit on error

# ========================================
# Configuration
# ========================================
# MODEL_PATH=""  # Model to train
MODEL_PATH=""  # Model to train
# MODEL_PATH=""
OUTPUT_DIR="outputs/tau2_training_inforl_qwen3-4b"
N_GPUS=4

export TMPDIR=""
export RAY_ENABLE_MEMORY_MONITOR=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
# Multi-turn interaction settings
export MULTITURN_MODEL_NAME="gpt-4o-mini-2024-07-18"
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
export TAU2_DATA_DIR=""
# Environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_MODE=offline
export LITELLM_LOG="ERROR"  # 只显示错误

# Suppress verbose logging from tau2-bench and Ray workers
export LOGURU_LEVEL="WARNING"  # Only show WARNING and ERROR logs from tau2-bench
export RAY_DEDUP_LOGS=1        # Deduplicate Ray logs to reduce noise

ulimit -n 65535

# Project directory
PROJECT_DIR=""
CONFIG_PATH="$PROJECT_DIR/examples/tau2/config"
DATA_DIR=""

echo "Installing/updating Tau2Gym package..."
pip install -e "$PROJECT_DIR/gyms/Tau2Gym" --quiet
echo "✓ Tau2Gym package updated"

# Dir for saving evaluation trajectories
export USERRL_EVAL_DUMP_DIR=""
mkdir -p "$USERRL_EVAL_DUMP_DIR"

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='tau2_trainer' \
    algorithm.adv_estimator=info_grpo \
    algorithm.use_intrinsic_reward=True \
    algorithm.gamma=0.8 \
    algorithm.use_kl_in_reward=False \
    \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.response_length_one_turn=1024 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=50 \
    actor_rollout_ref.rollout.multi_turn.model_name="${MULTITURN_MODEL_NAME}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${CONFIG_PATH}/tool_config/tau2_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="Equalized" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum" \
    trainer.critic_warmup=0 \
    trainer.resume_mode='disable' \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=1000 \
    trainer.val_before_train=False \
    trainer.total_epochs=10 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    $@

echo
echo "========================================="
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="
