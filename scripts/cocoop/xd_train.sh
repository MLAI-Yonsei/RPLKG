#!/bin/bash


# custom config
DATA=/DATA1/yewon/coop
TRAINER=ZeroshotCLIP
#TRAINER=CoCoOp
# TRAINER=CoOp

DATASET=imagenet
SEED=$1

#CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=vit_b16 #_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
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
    --mode "gumbel" \
    DATASET.NUM_SHOTS ${SHOTS}
fi