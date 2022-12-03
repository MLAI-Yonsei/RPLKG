
#!/bin/bash

# custom config



for DATASET in oxford_pets
do
    for SEED in 1 2
    do
        DATA=/DATA1/yewon/coop
        TRAINER=ZeroshotCLIP
        # TRAINER=CoOp
        #CFG=vit_b16_c4_ep10_batch1_ctxv1

        CFG=vit_b16  # uncomment this when TRAINER=CoOp
        #CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
        SHOTS=16


        DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        #if [ -d "$DIR" ]; then
        #    echo "Oops! The results exist at ${DIR} (so skip this job)"

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --mode "gumbel" \
        --dropout 0.4 \
        --wd 3e-3 \
        --alpha 0.2 \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    done
done
