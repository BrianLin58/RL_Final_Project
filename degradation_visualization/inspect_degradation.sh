#!/bin/bash

# choose: mild / medium / strong
level="medium"

python inspect_degradations.py \
  --image ../data/GOT10/train/GOT-10k_Train_000001/original/00000001.jpg \
  --cfg ../dataset/aug_${level}.yaml \
  --out ./degraded_images/inspect_${level}.jpg \
  --n 4
