# TS-CAM: Token Semantic Coupled Attention Map for Weakly SupervisedObject Localization
This is the official implementaion of paper [TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization](https://arxiv.org/abs/2103.14862)

This repository contains Pytorch training code, evaluation code, pretrained models and jupyter notebook for more visualization.

If you use this code for a paper please cite:

```
@article{Gao2021TSCAMTS,
  title={TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization},
  author={Wei Gao and Fang Wan and Xingjia Pan and Zhiliang Peng and Qi Tian and Zhenjun Han and Bolei Zhou and Qixiang Ye},
  journal={ArXiv},
  year={2021},
  volume={abs/2103.14862}
}
```

# Model Zoo

We provide pretrained TS-CAM models trained on CUB-200-2011 and ImageNet_ILSVRC2012 datasets.

| Dataset | Loc.Acc@1 | Loc.Acc@5 | Loc.Gt-Known | Cls.Acc@1 | Cls.Acc@5 | Baidu Drive | Google Drive |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  CUB-200-2011   |   71.3   |   83.8  |  87.7  |   80.3  |   94.8  |  [model](https://pan.baidu.com/s/1WdhcRh5pBFQD8DlbT_GoAQ)  | [model](https://drive.google.com/file/d/19l4uwsjE6uVah_0-a_VcRJlvnFb9NewH/view?usp=sharing) |
|  ILSVRC2012   |   53.4   |  64.3   |  67.6  |   74.3  |   92.1  |   [model](https://pan.baidu.com/s/11-iPVVtKvKpcfuOD8VwOZw)  | [model](https://drive.google.com/file/d/1iNH-zI2i9mGipF0rGo1lsp13avdIjWuS/view?usp=sharing) |

Note: the Extrate Code for Baidu Drive is as follows:
- CUB-200-2011: [36wz](https://pan.baidu.com/s/1WdhcRh5pBFQD8DlbT_GoAQ)
- ILSVRC2012:   [sslq](https://pan.baidu.com/s/11-iPVVtKvKpcfuOD8VwOZw)
 
# Usage

First clone the repository locally:
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

### CUB-200-2011 dataset

Please download and extrate [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset. 

The directory structure is the following:

```
TS-CAM/
  data/
    CUB-200-2011/
      attributes/
      images/
      parts/
      bounding_boxes.txt
      classes.txt
      image_class_labels.txt
      images.txt
      image_sizes.txt
      README
      train_test_split.txt
```

### ImageNet1k

Download [ILSVRC2012](http://image-net.org/) dataset and  extract train and val images.

The directory structure is organized as follows: 

```
TS-CAM/
  data/
  ImageNet_ILSVRC2012/
    ILSVRC2012_list/
    train/
      n01440764/
        n01440764_18.JPEG
        ...
      n01514859/
        n01514859_1.JPEG
        ...
    val/
      n01440764/
        ILSVRC2012_val_00000293.JPEG
        ...
      n01531178/
        ILSVRC2012_val_00000570.JPEG
        ...
    ILSVRC2012_list/
      train.txt
      val_folder.txt
      val_folder_new.txt
```

And the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

## For training:

On CUB-200-2011 dataset:
```
bash train_val_cub.sh {GPU_ID} ${NET}
```
On ImageNet1k dataset:
```
bash train_val_ilsvrc.sh {GPU_ID} ${NET}
```
Please note that pretrained model weights of Deit-tiny, Deit-small and Deit-base on ImageNet-1k model will be downloaded when you first train you model, so the Internet should be connected.

## For evaluation:
On CUB-200-2011 dataset:
```
bash val_cub.sh {GPU_ID} ${NET} ${MODEL_PATH}
```
On ImageNet1k dataset:
```
bash val_ilsvrc.sh {GPU_ID} ${NET} ${MODEL_PATH}
```
`GPU_ID` should be specified and multiple GPUs can be used for accelerating training and evaluation.

`NET` shoule be chosen among `tiny`, `small` and `base`.

`MODEL_PATH` is the path of pretrained model.

# Visualization
We provided `jupyter notebook` in `tools_cam` folder.
```
TS-CAM/
  tools-cam/
    visualization_attention_map_cub.ipynb
    visualization_attention_map_imaget.ipynb
```
Please download pretrained TS-CAM model weights and try more visualzation results((Attention maps using our method and [Attention Rollout](https://arxiv.org/abs/2005.00928) method)).
You can try other interseting images you like to show the localization map(ts-cams).

# Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails.
- [vasgaowei@gmail.com](vasgaowei@gmail.com)
- [qxye@ucas.ac.cn](qxye@ucas.ac.cn)
- [wanfang@ucas.ac.cn](wanfang@ucas.ac.cn)

You can also open an issue under this project.

