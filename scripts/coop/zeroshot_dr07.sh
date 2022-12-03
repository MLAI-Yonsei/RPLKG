for DATASET in caltech101 ucf101 stanford_cars oxford_pets oxford_flowers food101 fgvc_aircraft dtd eurosat sun397 imagenet
do
    for SHOTS in 16
    do
        DATA=/DATA1/yewon/coop
        TRAINER=ZeroshotCLIP
        CFG=rn50 # rn50, rn101, vit_b32 or vit_b16
        #caltech101 ucf101 stanford_cars oxford_pets oxford_flowers food101 fgvc_aircraft dtd eurosat sun397 imagenet
        SEED=1
        EMB_ROOT=/mlainas/KGPrompt_data
        SEARCH_LEVEL=1
        time=$(date +%F)_$(date +%H-%M)
        DIR=output/all/zs/${time}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/search_level${SEARCH_LEVEL}/seed${SEED}

        python train.py \
        --root ${DATA} \
        --dataset ${DATASET} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --mode "gumbel" \
        --report_name ZSCLIP_dataset_${DATASET} \
        --emb_root ${EMB_ROOT} \
        --search_level ${SEARCH_LEVEL} \
        --wd 3e-3 \
        --alpha 0.2 \
        --dropout 0.4 \
        DATASET.NUM_SHOTS ${SHOTS}
    done
done