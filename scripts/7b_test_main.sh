export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
python main_layerwise.py \
    --model_id  $MODEL \
    --save_dir ${SAVE_DIR}/${MODEL_ID}_$1_data/ \
    --start 0 \
    --end 1  \
    --num_gpus 1 \
    --dataset_name $1
