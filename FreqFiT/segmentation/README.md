
This repository is based on the official PyTorch implementation of [Towards Foundation Models and Few-Shot Parameter-Efficient Fine-Tuning for Volumetric Organ Segmentation](https://github.com/jusiro/fewshot-finetuning)

## 0. Installation
```
pip install 'monai[all]'
pip install -r requirements.txt
```

## 1. Datasets
Please download TotalSegmentator and place it inside ```local_data/datasets```.
* The employed train/test splits are located at  `local_data/partitions/transferability.txt`.
* Check `local_data/datasets/README.md` for an overview on how to organize these datasets.

## 2. Pre-trained models
Please, download the dataset from the weights link provided, and store it at models/pretrained_weights/[ID].pth for its use.

For reported results in our paper, we use pre-trained Swin-UNETR backbone [fseft.pth](https://drive.google.com/file/d/18yLNxmWGnVifQNeYYwyyu56Cg4tWV9aW/view?usp=sharing)

## FreqFit code
- The code for `FreqFit` method is in [models/architectures/SwinUNETR.py](https://github.com/tsly123/FreqFiT_medical/blob/main/segmentation/models/architectures/SwinUNETR.py#L1044)

- The code for integrating `FreqFit` into Swin-UNETR can be found in [models/architectures/SwinUNETR.py](https://github.com/tsly123/FreqFiT_medical/blob/main/segmentation/models/architectures/SwinUNETR.py#L870)


## 3. Parameter-Efficient Fine-tuning
Please see the [main_fseft.py](https://github.com/tsly123/FreqFiT_medical/blob/main/segmentation/main_fseft.py#L277) for fine-tuning options.

```
 python main_fseft.py --model_id fseft \
       --dataset totalseg --organ spleen \
       --k 1 \
       --method LoRA \
       --out_path ./out_path \
       --freqfit True \
       --decoder frozen \
       --lr 1e-4 \
       --seeds 3
```
