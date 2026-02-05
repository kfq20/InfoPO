#!/bin/bash

# Required field
export MAX_WORKER_NUM=4 # Maximum number of workers to use
export USER_MODEL_NAME="gpt-4o-mini-2024-07-18" # Model name for the user
export OPENAI_API_KEY=""
export OPENAI_BASE_URL="" # Base URL for the OpenAI API
export TOOL_CHOICE="auto" # Tool choice for the model, please keep it as "auto"
export PROJECT_ROOT=""

# API key for the Gemini API
export GENAI_API_KEY="XXX"
# If using SearchGym to test on bamboogle
export SERPER_API_KEY=""
export SERPER_BASE_URL=""

# If you are using the official model
# python eval.py \
#     --model_name qwen2.5-7b-instruct \
#     --max_turns 16 \
#     --pass_k 1 \
#     --temperature 0 \
#     --envs bamboogle \
#     --save_name outputs/results_4omini