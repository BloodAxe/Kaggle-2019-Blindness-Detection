#!/usr/bin/env bash

python train_classifier_baseline.py -m cls_resnext50_hyp -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -v --use-aptos2019 --use-idrid --use-messidor --warmup 2
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -v --use-aptos2019 --use-idrid --use-messidor --warmup 2
sleep 5

python train_regression_baseline.py -m reg_resnext101_rms -a medium -f 0 -b 48 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -v --use-aptos2019 --use-idrid --use-messidor --warmup 2
sleep 5
