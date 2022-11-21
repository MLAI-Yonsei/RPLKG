#!/bin/bash
cd ../..
# custom config
DATA=/disk/changdae/data_coop
#DATA=/disk/changdae/data_coop
TRAINER=ZeroshotCLIP
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16

SEED=1
SHOTS=16
#caltech101 ucf101 stanford_cars oxford_pets oxford_flowers food101 fgvc_aircraft dtd eurosat sun397 imagenet
for DATASET in imagenet
do

DIR=output/all/zs/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

CUDA_VISIBLE_DEVICES=7 /disk/teang1995/anaconda3/envs/coop/bin/python3 train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
--eval-only \
--mode "attention" \
--report_name ZSCLIP_dataset_${DATASET} \
DATASET.SUBSAMPLE_CLASSES all

done
