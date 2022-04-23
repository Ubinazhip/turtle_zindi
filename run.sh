#!/bin/sh


python3 train.py --model_type b3 --batch_size 4 --epochs 50 --fold 0
python3 train.py --model_type b3 --batch_size 4 --epochs 50 --fold 1
python3 train.py --model_type b3 --batch_size 4 --epochs 50 --fold 2



python3 train.py --model_type b4 --batch_size 4 --epochs 50 --fold 2
python3 train.py --model_type b4 --batch_size 4 --epochs 50 --fold 3
python3 train.py --model_type b4 --batch_size 4 --epochs 50 --fold 4


python3 train.py --model_type b5 --batch_size 4 --epochs 50 --fold 0
python3 train.py --model_type b5 --batch_size 4 --epochs 50 --fold 2
python3 train.py --model_type b5 --batch_size 4 --epochs 50 --fold 4

python3 train.py --model_type b6 --batch_size 4 --epochs 50 --fold 3
python3 train.py --model_type b6 --batch_size 4 --epochs 50 --fold 4


CUDA_VISIBLE_DEVICES=2,3 python3 train.py --model_type b7 --batch_size 4 --epochs 50 --fold 0

#CUDA_VISIBLE_DEVICES=0,1 python3 inference.py --model_type b8 --batch_size 16 --num_classes 101 --model_path /home/ibm_prod/breast/model/backbone/code/b8_fold3_sampler.pth



