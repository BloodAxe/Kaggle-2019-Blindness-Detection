#!/usr/bin/env bash

python train_regression_baseline.py -m reg_inceptionv4_rms -a hard -f 0 -f 1 -f 2 -f 3 -b 60 -l huber --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019 --warmup 20
sleep 5