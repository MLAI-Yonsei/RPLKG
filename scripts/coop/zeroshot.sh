#!/bin/bash

# custom config
DATA=/data/yewon/DATA
TRAINER=ZeroshotCLIP
DATASET=imagenet
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
SHOTS=16
time=$(date +%F)_$(date +%H-%M)
path=output/zs_gumbel_im_test_1/${time}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
mkdir -p $path
DIR=$path

echo $path
python train.py \
--root ${DATA} \
--seed 1 \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
--mode "attention" \
--dropout 0.1 \
--wd 0.02 \
--report_name ZSCLIP_dataset_${DATASET} \
DATASET.NUM_SHOTS ${SHOTS} \

#--anneal_epoch
#--max_temp
#--min_temp