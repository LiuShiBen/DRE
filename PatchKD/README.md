# PatchKD
Code for ACM MM 2022 paper [Patch-based Knowledge Distillation for Lifelong Person Re-Identification](http://www.muyadong.com/paper/acmmm22_sunzc.pdf).

![Framework](figs/framework.png)

## Installation
```shell
git clone https://github.com/feifeiobama/PatchKD
cd PatchKD
pip install -r requirements.txt
python setup.py develop
```

Please follow [Torchreid_Datasets_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) to download datasets and unzip them to your data path (we refer to 'machine_dataset_path' in train_test.py). Alternatively, you could download some datasets from [light-reid](https://github.com/wangguanan/light-reid) and [DualNorm](https://github.com/BJTUJia/person_reID_DualNorm).

## Quick Start
Training + evaluation. Make sure the visdom server is listening.
```shell
python train_test.py
```

Evaluation from checkpoint:
```shell
python train_test.py --mode test --resume_test_model /path/to/pretrained/model
```

Visualization from checkpoint:
```shell
python train_test.py --mode visualize --resume_visualize_model /path/to/pretrained/model
```

Experiment under a different training order:
```shell
python train_test.py --train_dataset duke msmt17 market subcuhksysu cuhk03
```

## Results
The following results were obtained with single NVIDIA 2080 Ti GPU:

![Results](figs/results.png)

## Citation
If you find this code useful for your research, please cite our paper.
```
@inproceedings{sun2022patch,
    author = {Sun, Zhicheng and Mu, Yadong},
    title = {Patch-based Knowledge Distillation for Lifelong Person Re-Identification},
    booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
    pages = {696--707},
    year = {2022}
}
```

## Acknowledgement
Our code is based on the PyTorch implementation of [LifelongReID](https://github.com/TPCD/LifelongReID).
