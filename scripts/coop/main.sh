#!/bin/bash

# custom config

for SHOTS in 8
do
    for DATASET in imagenet # oxford_flowers food101 fgvc_aircraft dtd eurosat sun397 caltech101 ucf101 stanford_cars 
    do
        DIR=/DATA1/yewon/output_coop/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        DATA=/DATA1/yewon/coop
        TRAINER=CoOp
        CFG=vit_b16  # config file
        CTP=end  # class token position (end or middle)
        NCTX=16  # number of context tokens # number of shots (1, 2, 4, 8, 16)
        CSC=False  # class-specific context (False or True)
        SEED=1
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}_.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done