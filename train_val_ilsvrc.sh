GPU_ID=$1
NET=${2}
NET_SCALE=${3}
SIZE=${4}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/${NET}_tscam_${NET_SCALE}_patch16_${SIZE}.yaml --lr 5e-4 MODEL.CAM_THR 0.12
