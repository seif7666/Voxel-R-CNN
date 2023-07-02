#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# set -x
NGPUS=1
CFG_DIR=cfgs

CFG_NAME=voxel_rcnn/voxel_rcnn_car

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --epochs 1 --workers 2 --pretrained_model "/content/drive/MyDrive/Graduation Project/output3/ckpt/checkpoint_epoch_1.pth"
