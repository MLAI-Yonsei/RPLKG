# RPLKG: Robust Prompt Learning with Knowledge Graph
- Yewon Kim, YongTaek Lim, Dokyung Yoon, KyungWoo Song. 
- This repo contains official implementation of [RPLKG: Robust Prompt Learning with Knowledge Graph](https://arxiv.org/pdf/2304.10805) .

## Requirement and Datasets
Our code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated).

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run
You can implement our code by running this command. Change dataset and hyperparmeters by following arguments
```bash
python train.py  --root ${DATA}  --seed 1 --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} --mode "gumbel" --dropout ${dropout} --wd ${wd} --logit_scale ${scale} 
```

## Copyright
- Our work based on Learning to Prompt for Vision-Language Models([CoOp](https://github.com/KaiyangZhou/CoOp))
```bash
@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}

@article{zhou2022coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={International Journal of Computer Vision (IJCV)},
    year={2022}
}
```
