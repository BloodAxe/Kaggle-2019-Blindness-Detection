#!/usr/bin/env bash

python train_regression_baseline.py -m seresnext50d_gwap -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4
python train_regression_baseline.py -m seresnext50d_gap  -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4