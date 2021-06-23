GPU_ID=$1
NET=${2}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/deit_tscam_${NET}_patch16_224.yaml --lr 5e-4 MODEL.CAM_THR 0.12
