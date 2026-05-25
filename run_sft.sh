export OMP_NUM_THREADS=16
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

export OUTPUT_DIR=output_sft_v1
export TENSORBOARD_LOGGING_DIR=$OUTPUT_DIR/runs

accelerate launch --config_file accelerate_config.yaml main_sft_v1.py \
  --output-dir $OUTPUT_DIR \
  --logging-dir $TENSORBOARD_LOGGING_DIR \
  --tokenizer-path minimind \
  --train-text-file dataset/minimind/sft_t2t_mini.jsonl \
  --model-path output_v1_1/checkpoint-2000 \

# uv run tensorboard --logdir $TENSORBOARD_LOGGING_DIR --port 6006
