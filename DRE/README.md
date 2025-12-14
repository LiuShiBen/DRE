## Diverse Representations Embedding for Lifelong Person Re-Identification (DRE)

<div align="center"> 

🎯Our paper has been accepted by [IEEE Transactions on Neural Networks and Learning Systems 2025](https://ieeexplore.ieee.org/document/11045423)  
</div>

## Introduction
```
Our work proposes an “Diverse Representations Embedding for Lifelong Person Re-Identification” (DRE). The proposed DRE adaptively learns diverse representation to achieve a dynamic balance between preserving old knowledge and adapting to new information.
```
![](./DRE/docs/DRE.png)

### Requirements
- Python 3.8
- Pytorch 1.7.0
- For more detailed requirements, run
```
pip install -r requirements.txt
```
## Dataset preparation
- Please follow [Torchreid_Datasets_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) to download datasets and unzip them to your data path .
- Prepare the Seen dataset structure as follow:   ./docs/seen dataset_structure .md
- Prepare the Unseen dataset structure as follow:  ./docs/Unseen dataset_structure .md
### Prepare ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).

## Training

- Training order-1  

```
python continual_train(order1).py -data-dir=/your seen dataset path  --logs-dir==/save path
```

- Training order-2

```
python continual_train(order2).py -data-dir=/your seen dataset path  --logs-dir==/save path
```

## Testing
```
python evaluate.py --data-dir=/your test dataset path"
```

## Acknowledgement

Thanks for all these great code bases:
- The code framework is based on [KRKC](https://github.com/cly234/LReID-KRKC).

## Contact
If you have any questions, please contact Shiben Liu at liushiben310@163.com.
