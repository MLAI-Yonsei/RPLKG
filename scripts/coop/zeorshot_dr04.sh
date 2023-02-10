# custom config
for SHOTS in 1
do
    for SEARCH_LEVEL in  4
    do
        for DATASET in eurosat #caltech101 ucf101 stanford_cars oxford_pets oxford_flowers food101 fgvc_aircraft dtd eurosat sun397 imagenet
        do
            
            DATA=/DATA1/yewon/coop
            #DATA=/data3/yewon/data_coop
            TRAINER=ZeroshotCLIP
            CFG=vit_b16 # rn50, rn101, vit_b32 or vit_b16
            SEED=1
            EMB_ROOT=/mlainas/KGPrompt_data
            time=$(date +%F)_$(date +%H-%M)
            DIR=output_ours/${CFG}/_search_level${SEARCH_LEVEL}/${DATASET}__/shots_${SHOTS}/${TRAINER}/seed${SEED}

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
done