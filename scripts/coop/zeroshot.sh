#!/bin/bash

# custom config

for alpha in 0
do
    DATA=/DATA1/yewon/coop
    TRAINER=ZeroshotCLIP
    DATASET=imagenet
    CFG=vit_b16 
    SHOTS=16
    time=$(date +%F)_$(date +%H-%M)
    path=output/zs_gumbel_im_test_1/${time}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    mkdir -p $path
    DIR=$path
    echo $path

    python train.py  --wd 3e-3 --dropout 0.4 --alpha ${alpha} --root ${DATA}  --seed 1 --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} --mode "gumbel" --report_name ZSCLIP_dataset_${DATASET} DATASET.NUM_SHOTS ${SHOTS} 

done
