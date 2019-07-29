#!/usr/bin/env bash

python train_classifier_baseline.py -m reg_seresnext50_max -a hard -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019
sleep 5

python train_regression_baseline.py -m reg_seresnext50_max -a hard -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019
sleep 5
