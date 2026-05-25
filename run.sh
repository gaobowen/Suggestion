export OMP_NUM_THREADS=16
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

export OUTPUT_DIR=output_v1_1
export LOGGING_DIR=$OUTPUT_DIR/runs

accelerate launch --config_file accelerate_config.yaml main_pretrain_v1.py \
--output-dir $OUTPUT_DIR \
--logging-dir $LOGGING_DIR \
--tokenizer-path minimind \
--text-file dataset/minimind/pretrain_t2t_mini.jsonl \
--simple-data-pipeline

# uv run tensorboard --logdir $LOGGING_DIR --port 6006