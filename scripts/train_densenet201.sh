#!/usr/bin/env bash

python train_ord.py -m densenet201_gwap  -a hard -f 0 -f 1 -f 2 -f 3 -b 18 -d 0.25 --size 512 -o SGD -wd 1e-4 --warmup 10 -s multistep -lr 3e-4 -e 75 --use-idrid --use-aptos2019 -v  --criterion-cls focal_kappa 0.25 --criterion-reg mse 0.25