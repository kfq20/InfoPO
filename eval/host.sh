export CUDA_VISIBLE_DEVICES=4,5

export MERGED_MODEL_PATH=""
export CUSTOMIZED_SERVED_MODEL_NAME="infopo-qwen3-4b"

vllm serve ${MERGED_MODEL_PATH} \
   --max-model-len 32768 \
   --port 8500 \
   --gpu-memory-utilization 0.7 \
   --tensor-parallel-size 2 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --served-model-name ${CUSTOMIZED_SERVED_MODEL_NAME}
   # --reasoning-parser qwen3 \
