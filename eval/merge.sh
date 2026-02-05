

MODEL_PATH=""

python merge.py merge \
    --backend fsdp \
    --local_dir ${MODEL_PATH}/actor \
    --target_dir ${MODEL_PATH}_hf
