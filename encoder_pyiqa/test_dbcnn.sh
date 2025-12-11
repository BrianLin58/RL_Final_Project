#!/bin/bash

cd /home/betty/RL_Final_Project/encoder_pyiqa

python test_dbcnn.py \
  --image ../data/GOT10/train/GOT-10k_Train_000001/degraded/00000002.jpg

python test_dbcnn.py \
  --image ../data/GOT10/train/GOT-10k_Train_000001/original/00000002.jpg