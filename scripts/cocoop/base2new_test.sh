#!/bin/bash


# custom config

DATA=/DATA1/yewon/coop
#TRAINER=CoCoOp
# TRAINER=CoOp
TRAINER=ZeroshotCLIP

#CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=vit_b16 #_ctxv1  # uncomment this when TRAINER=CoOp
#CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and imagenet

SHOTS=16
LOADEP=100
SUB=new

DATASET=$1
SEED=$2


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}/
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
   echo "Oops! The results exist at ${DIR} (so skip this job)"
else
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
--mode 'attention' \
--alpha 3 \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
fi

