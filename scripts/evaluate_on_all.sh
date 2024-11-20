#!/bin/bash
model_path=$1
results_folder=$2
gpu_device=$3
dataset_cache_dir=$4
root_dataset_path=$5

mkdir -p "${results_folder}"

# VSR
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_vsr.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--split test \
	--output_json "${results_folder}"/vsr_test.json

# AI2D
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_ai2d.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--output_json "${results_folder}"/ai2d.json

# COCO captioning
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_coco_captioning.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--model_max_length 100 \
	--max_new_tokens 40 \
	--split test \
	--prediction_json "${results_folder}"/coco_test_predictions.json \
	--groundtruth_json "${results_folder}"/coco_test_groundtruths.json \
	--metric_json "${results_folder}"/coco_test_metrics.json

# TextCaps
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_textcaps.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--model_max_length 100 \
	--max_new_tokens 15 \
	--split validation \
	--prediction_json "${results_folder}"/textcaps_validation_predictions.json \
	--groundtruth_json "${results_folder}"/textcaps_validation_groundtruths.json \
	--metric_json "${results_folder}"/textcaps_validation_metrics.json

#VQAav2
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_vqav2.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--model_max_length 100 \
	--max_new_tokens 15 \
	--split validation \
	--output_json "${results_folder}"/vqa_v2_validation.json

# GQA
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_gqa.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--model_max_length 100 \
	--output_json "${results_folder}"/gqa_testdev.json \
	--split testdev \
	--max_new_tokens 20

# RefCOCOg
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_refcoco.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--eval_dataset_subset refcocog \
	--split test \
	--output_json "${results_folder}"/refcocog_test.json

CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_refcoco.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--eval_dataset_subset refcoco+ \
	--split testA \
	--output_json "${results_folder}"/refcoco+_testA.json

CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_refcoco.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--eval_dataset_subset refcoco+ \
	--split testB \
	--output_json "${results_folder}"/refcoco+_testB.json

CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_refcoco.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--eval_dataset_subset refcoco \
	--split testA \
	--output_json "${results_folder}"/refcoco_testA.json

CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_refcoco.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--eval_dataset_subset refcoco \
	--split testB \
	--output_json "${results_folder}"/refcoco_testB.json

# POPE
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_pope.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--output_json "${results_folder}"/pope_popular.json \
	--eval_dataset_subset pope_popular \
	--max_length 1

CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_pope.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--output_json "${results_folder}"/pope_random.json \
	--eval_dataset_subset pope_random \
	--max_length 1

CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_pope.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--output_json "${results_folder}"/pope_adversarial.json \
	--eval_dataset_subset pope_adversarial \
	--max_length 1

# NoCaps
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_nocaps.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--max_new_tokens 40 \
	--prediction_json "${results_folder}"/nocaps_validation_predictions.json \
	--groundtruth_json "${results_folder}"/nocaps_validation_groundtruths.json \
	--metric_json "${results_folder}"/nocaps_validation_metrics.json \
	--split validation

# Visual7W
CUDA_VISIBLE_DEVICES=${gpu_device} python src/vl_mamba/evaluation/evaluate_visual7w.py \
	--dataset_cache_dir "${dataset_cache_dir}" \
	--root_dataset_path "${root_dataset_path}" \
	--model_name "${model_path}" \
	--model_max_length 100 \
	--max_new_tokens 20 \
	--split test \
	--output_json "${results_folder}"/visual7w.json
