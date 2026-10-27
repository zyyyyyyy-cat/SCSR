# DDP
## WHU-MARS-1000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_1000_DDP

CUDA_VISIBLE_DEVICES=7 python test.py \
  --config_file configs/UAD.yml \
  DATASETS.DATASET_DIR WHU-MARS \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_DDP

CUDA_VISIBLE_DEVICES=7 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True \
  OUTPUT_DIR logs/UAD_1000_DDP_domain_eval

## WHU-MARS-1000-GD
CUDA_VISIBLE_DEVICES=6 python test.py \
  --config_file configs/UAD.yml \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_GD

## WHU-MARS-2337
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_2337_DDP

# Single
## WHU-MARS-1000

cd /project/code/WHU_base/WHU_DDP

CUDA_VISIBLE_DEVICES=7 python train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 1024 \
  OUTPUT_DIR logs/UAD_1000_single2

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  TEST.IMS_PER_BATCH 1024 \
  TEST.WEIGHT logs/UAD_1000_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_single

## WHU-MARS-1000-GD

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_single_GD

## WHU-MARS-2337

CUDA_VISIBLE_DEVICES=6 python train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 1024 \
  OUTPUT_DIR logs/UAD_2337_single

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  TEST.IMS_PER_BATCH 1024 \
  TEST.WEIGHT logs/UAD_2337_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_2337_single