### Train/Test


### This code is based on pytorch1.7. And before running the codes, please follow several steps to install the environment and prepare the datasets.

### Run the following command to install the development environment:

conda create -n pytorch1.7 python=3.6
conda activate pytorc1.7
conda install anaconda
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.3.2 

### For training:
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ./tools_cam/train_cam.py --config_file ./configs/CUB/deit_cam_small_patch16_224.ymal --lr 5e-5 MODEL.CAM_THR 0.1 

### Note that training and testing are done alternately, it means that when model are trained for one epoch then model is tested immediately.
### Please keep other config parameters as default when try to re-produce the results in paper. Of course, you can choose different config file or different learning rate.

### For testing with localization maps saved, please run the following commands and set TEST.SAVE_BOXED_IMAGE True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ./tools_cam/test_cam.py --config_file configs/CUB/deit_cam_small_patch16_224.ymal --resume save_path TEST.SAVE_BOXED_IMAGE True MODEL.CAM_THR 0.1


