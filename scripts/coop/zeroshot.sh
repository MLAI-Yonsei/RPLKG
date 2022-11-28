#!/bin/bash

# custom config

for dropout in 0.4 
do
    for alpha in 0.2
    do
        for wd in 3e-3
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

            python train.py --alpha ${alpha} --root ${DATA} --max_temp 10 --min_temp 0.001 --seed 1 --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} --mode "gumbel" --dropout ${dropout} --wd ${wd} --report_name ZSCLIP_dataset_${DATASET} DATASET.NUM_SHOTS ${SHOTS}
        
        done
    done
done

