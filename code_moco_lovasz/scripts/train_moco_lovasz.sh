#!/bin/bash
python main_moco.py \
  -a resnet18 \
  --lr 0.03 \
  --batch-size 256 \
  --moco-k 8192 \
  --method moco_lovasz \
  --sim-mat /path/to/similarities \
  --no-stability \
  --dist-url 'tcp://localhost:' --multiprocessing-distributed --world-size 1 --rank 0 \
 /path/to/data/ \
 name