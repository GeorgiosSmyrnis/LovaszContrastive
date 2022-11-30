#!/bin/bash
python main_lincls.py \
  -a resnet18 \
  --pretrained /path/to/ckpt \
  --dist-url 'tcp://localhost:' --multiprocessing-distributed --world-size 1 --rank 0 \
 /path/to/data \
 name
