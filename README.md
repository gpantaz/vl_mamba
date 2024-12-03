# Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision \& Language Modeling (EMNLP 2024)
[[Paper](https://arxiv.org/pdf/2409.05395)][[Model Checkpoints](#model-checkpoints)][[Data](#data)][[Training](#training)]

## Requirements

```
conda create -p ~/sharedscratch/conda_envs/athnlp2024-cu12 python=3.10

# Install dependencies
poetry install

# Install flash attention / mamba

poe install-flash-attn
poe install-mamba-ssm
poe install-causal-conv1d
```


## Model Checkpoints 🤗


### Pretrained Checkpoints


| Model                                                                     |
| :------------------------------------------------------------------------ |
| [Pythia-VL-1B](https://huggingface.co/gpantaz/pretrained_pythiavl_1b)     |
| [Mamba-VL-790M](https://huggingface.co/gpantaz/pretrained_mambavl_790m)   |
| [Pythia-VL-1.4B](https://huggingface.co/gpantaz/pretrained_pythiavl_1.4b) |
| [Mamba-VL-1.4B](https://huggingface.co/gpantaz/pretrained_mambavl_1.4b)   |
| [Pythia-VL-2.8B](https://huggingface.co/gpantaz/pretrained_pythiavl_2.8b) |
| [Mamba-VL-2.8B](https://huggingface.co/gpantaz/pretrained_mambavl_2.8b)   |

### Instruction-tuned Checkpoints

| Model                                                                    |  COCO  | NoCaps | VQAv2 |  GQA  | V7W (test-T) |  VSR  | POPE  | RefCOCO (testA) | RefCOCO (testB) | RefCOCO+ (testA) | RefCOCO+ (testB) | RefCOCOg | V7W (test-P) | TextCaps | TextVQA | AI2D  |
| :----------------------------------------------------------------------- | :----: | :----: | :---: | :---: | :----------: | :---: | :---: | :-------------: | :-------------: | :--------------: | :--------------: | :------: | :----------: | :------: | :-----: | :---: |
|                                                                          |        |        |       |       |              |       |       |                 |                 |                  |                  |          |              |          |         |       |
| [Pythia-VL-1B](https://huggingface.co/gpantaz/finetuned_pythiavl_1b)     | 132.89 | 97.61  | 72.26 | 53.79 |    81.96     | 72.43 | 86.77 |      76.00      |      62.48      |      45.36       |      47.44       |  67.58   |    83.78     |  92.73   |  35.22  | 77.62 |
| [Mamba-VL-790M](https://huggingface.co/gpantaz/finetuned_mambavl_790m)   | 133.81 | 99.00  | 71.67 | 54.95 |    81.82     | 75.39 | 86.77 |      67.84      |      56.35      |      57.97       |      41.43       |  59.16   |    74.01     |  94.30   |  40.72  | 79.27 |
| [Pythia-VL-1.4B](https://huggingface.co/gpantaz/finetuned_pythiavl_1.4b) | 134.06 | 100.72 | 73.57 | 57.05 |    83.06     | 77.72 | 86.40 |      82.43      |      68.39      |      72.35       |      55.16       |  72.56   |    86.13     |  94.60   |  37.54  | 79.27 |
| [Mamba-VL-1.4B](https://huggingface.co/gpantaz/finetuned_mambavl_1.4b)   | 134.76 | 100.56 | 74.46 | 58.44 |    83.78     | 80.18 | 85.32 |      76.60      |      63.48      |      68.40       |      52.11       |  68.82   |    80.18     |  98.68   |  41.30  | 80.86 |
| [Pythia-VL-2.8B](https://huggingface.co/gpantaz/finetuned_pythiavl_2.8b) | 134.97 | 101.27 | 75.08 | 59.76 |    84.34     | 80.86 | 86.87 |      85.39      |      70.82      |      75.39       |      58.62       |  76.24   |    86.61     |  99.74   |  39.14  | 81.57 |
| [Mamba-VL-2.8B](https://huggingface.co/gpantaz/finetuned_mambavl_2.8b)   | 135.53 | 102.00 | 76.08 | 60.41 |    85.31     | 81.45 | 87.33 |      79.29      |      64.97      |      71.64       |      53.94       |  71.27   |    82.50     |  100.47  |  42.14  | 83.71 |


## Data

### Instructions for specific datasets

#### GRIT

GRIT is downloaded using [img2dataset](https://github.com/rom1504/img2dataset). Note that some of
the urls may not be available by the time of the downloading

```
./scripts/download_grit.sh storage/datasets/grit_url_folder storage/datasets/grit
```


To avoid training on the whole data, filter out grit by the noun_phrases (see appendix in the paper
for full details)
```
python prepare_grit_dataset.py \
	--cache_dir /path/to/downloaded/grit \
	--output_folder /path/to/downsampled/grit \
	--downsample_images \
	--check_overlap 
```

#### OCRVQA

We also filter out examples from OCRVQA (see appendix in the paper for details)

```
python filter_out_ocrvqa_images.py \
	--cache_dir /path/to/downloaded/ocrvqa \
	--output_json /path/to/filtered/ocrvqa/examples \
```


### Prepare pretraining dataset

```
python prepare_dataset.py \
	--dataset_subset llava_pretrain \
	--root_dataset_path storage/datasets \
	--cache_dir storage/datasets/vl_mamba \
```


### Prepare instruction tuning dataset
```
python prepare_dataset.py \
	--dataset_subset instruction_tuning \
	--root_dataset_path storage/datasets \
	--cache_dir storage/datasets/vl_mamba \
```


### Prepare a single dataset
```
python prepare_dataset.py \
	--dataset_subset coco \
	--root_dataset_path storage/datasets \
	--cache_dir storage/datasets/vl_mamba \
```

see `DatasetNames` in `src/vl_mamba/datamodels/datamodels.py` for the names of different datasets

## Training

### Pretraining

#### Pythia

```
./scripts/pretrain_pythia.sh
```

#### Mamba
```
./scripts/pretrain_mamba.sh
```

### Instruction-tuning

#### Pythia

```
./scripts/finetune_pythia.sh path/to/pretrained/pythia/model /path/to/dataset/cache /path/to/root/dataset/path /output/model/directory wandb_run_name
```

#### Mamba
```
./scripts/finetune_mamba.sh path/to/pretrained/mamba/model /path/to/dataset/cache /path/to/root/dataset/path /output/model/directory wandb_run_name
```

### Training logs

All the logs regarding pretraining / finetuning can be found on [wandb](https://wandb.ai/gpantaz/vl_mamba?nw=nwusergpantaz)
Note that some of the runs were resumed from a previous checkpoint.
