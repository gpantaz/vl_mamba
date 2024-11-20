url_folder=$1
output_image_folder=$2

mkdir -p $url_folder
mkdir -p $output_image_folder

# Download the .parquet files
# For some reason the git clone command does not work
for idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
	if [ -f "${url_folder}/coyo_${idx}_snappy.parquet" ]; then
		echo "File ${url_folder}/coyo_${idx}_snappy.parquet exists."
	else
		wget https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet -O ${url_folder}/coyo_${idx}_snappy.parquet
	fi
done

img2dataset --url_list $1 --input_format "parquet" \
	--url_col "url" --caption_col "caption" \
	--output_folder $2 --processes_count 16 --thread_count 64 --image_size 256 \
	--resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
	--save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
	--enable_wandb False --incremental
