#!/bin/bash
pretrained_model_path=$1
dataset_cache_dir=$2
root_dataset_path=$3
output_dir=$4
run_name=$5

python src/vl_mamba/train_hf.py \
	--model_name "${pretrained_model_path}" \
	--vision_encoder_name timm/eva02_large_patch14_clip_336 \
	--pixel_based False \
	--train_only_visual_embeddings False \
	--image_size 336 \
	--patch_size 14 \
	--num_channels 3 \
	--model_max_length 1024 \
	--variable_sized False \
	--pixel_based False \
	--box_mode normalize \
	--dataset_path src/vl_mamba/datasets/vl_mamba \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--train_dataset_subset instruction_tuning \
	--eval_dataset_subset instruction_tuning \
	--output_dir "${output_dir}" \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 2 \
	--logging_steps 1 \
	--save_strategy steps \
	--save_steps 0.1 \
	--num_train_epochs 1 \
	--learning_rate 2e-5 \
	--weight_decay 0. \
	--warmup_ratio 0.03 \
	--lr_scheduler_type cosine \
	--bf16 True \
	--fp16 False \
	--gradient_checkpointing False \
	--deepspeed configs/trainer/zero3.json \
	--save_total_limit 2 \
	--load_best_model_at_end False \
	--log_level info \
	--save_safetensors True \
	--evaluation_strategy steps \
	--eval_steps 0.1 \
	--seed 12345 \
	--data_seed 12345 \
	--dataloader_num_workers 4 \
	--logging_nan_inf_filter False \
	--run_name "${run_name}" \
	--project_name vl-mamba \
	--report_to wandb
