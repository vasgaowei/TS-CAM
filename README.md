# TS-CAM: Token Semantic Coupled Attention Map for Weakly SupervisedObject Localization
This is the official implementaion of paper [TS-CAM: Token Semantic Coupled Attention Map for Weakly SupervisedObject Localization](https://arxiv.org/abs/2103.14862)
This repository contains Pytorch training code, evaluation code, pretrained models and jupyter notebook for more visualization.

# Usage

Firstm clone the repository locally:
```
git clone https://github.com/vasgaowei/TS-CAM.git
```
Then install Pytorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):


```

conda create -n pytorch1.7 python=3.6
conda activate pytorc1.7
conda install anaconda
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## For training:

On CUB-200-2011 dataset:
```
bash train_val_cub.sh {GPU_ID} ${NET}
```
On ImageNet1k dataset:
```
bash train_val_ilsvrc.sh {GPU_ID} ${NET}
```

## For evaluation:
On CUB-200-2011 dataset:
```
bash val_cub.sh {GPU_ID} ${NET} ${MODEL_PATH}
```
On ImageNet1k dataset:
```
bash val_ilsvrc.sh {GPU_ID} ${NET} ${MODEL_PATH}
```
