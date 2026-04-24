export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0

accelerate launch --config_file accelerate_config.yaml main_pretrain.py

# uv run tensorboard --logdir output/runs --port 6006