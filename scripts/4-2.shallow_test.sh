#!/bin/bash

dataset="mimic_cxr"
annotation="data/mimic_cxr/my_mimic_anno.json"
base_dir="./data/mimic_cxr/images"
delta_file="path/to/pretrained/delta_file"

version="v1_shallow"
savepath="./save/$dataset/$version"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt