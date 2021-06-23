GPU_ID=$1
NET=${2}
PATH=${3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/test_cam.py --config_file configs/ILSVRC/deit_tscam_${NET}_patch16_224.yaml --resume ${PATH} TEST.SAVE_BOXED_IMAGE True MODEL.CAM_THR 0.12
