#!/bin/bash

dataset="mimic_cxr"
annotation="data/mimic_cxr/my_mimic_anno.json"
base_dir="./data/mimic_cxr/images"

version="v0_shallow"
savepath="./save/$dataset/$version"

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 16 \
    --val_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 3 \
    2>&1 |tee -a ${savepath}/log.txt