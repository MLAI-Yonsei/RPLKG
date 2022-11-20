#!/bin/bash

# custom config
DATA=/data/yewon/DATA
#DATA=/disk/changdae/data_coop
TRAINER= ZeroshotCLIP
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
#caltech101 ucf101 stanford_cars oxford_pets oxford_flowers food101 fgvc_aircraft dtd eurosat sun397 imagenet
DATASET=imagenet 

SEED=1
SHOTS=16

DIR=output/all/zs/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py  \

--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
--mode "attention" \
--report_name ZSCLIP_dataset_${DATASET} \
#--eval-only \




DATASET.NUM_SHOTS ${SHOTS}

