#!/bin/bash
# Training script for ColBench with UserRL
# This script demonstrates how to train an RL agent on ColBench code collaboration tasks

set -e  # Exit on error

# ========================================
# Configuration
# ========================================
MODEL_PATH=""
OUTPUT_DIR="outputs/colbench_training"
N_GPUS=4

# Temporary directory for Ray
export TMPDIR=""
mkdir -p $TMPDIR

# Multi-turn interaction settings
# The model used to simulate human collaborator in ColBench
export MULTITURN_MODEL_NAME="gpt-4o-mini-2024-07-18"
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""
# Environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

ulimit -n 65535

# Qwen3 "thinking" sanitization
# - USERRL_STRIP_QWEN_THINK_OUTPUT=1: strip any accidental <think>...</think> blocks from generated text before env/reward
export USERRL_STRIP_QWEN_THINK_OUTPUT=${USERRL_STRIP_QWEN_THINK_OUTPUT:-1}

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/colbench/config"
DATA_DIR="$PROJECT_DIR/data/colbench_code"

echo "========================================="
echo "ColBench Training Setup"
echo "========================================="
echo "Project Directory: $PROJECT_DIR"
echo "Config Path: $CONFIG_PATH"
echo "Data Directory: $DATA_DIR"
echo "Model Path: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $N_GPUS"
echo "========================================="

# Install/update ColBenchGym package
echo "Installing/updating ColBenchGym package..."
pip install -e "$PROJECT_DIR/gyms/ColBenchGym" --quiet
echo "✓ ColBenchGym package updated"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Dir for saving evaluation trajectories
export USERRL_EVAL_DUMP_DIR="${OUTPUT_DIR}/eval_logs"
mkdir -p "$USERRL_EVAL_DUMP_DIR"

echo "Starting training..."
echo "========================================="

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='colbench_trainer' \
    algorithm.adv_estimator=info_grpo \
    algorithm.use_intrinsic_reward=True \
    algorithm.gamma=0.8 \
    algorithm.use_kl_in_reward=False \
    \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
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
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=10 \
    actor_rollout_ref.rollout.multi_turn.model_name="${MULTITURN_MODEL_NAME}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${CONFIG_PATH}/tool_config/colbench_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="Equalized" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.resume_mode="disable" \
    trainer.save_freq=250 \
    trainer.test_freq=10000 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    $@

echo
echo "========================================="
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="
