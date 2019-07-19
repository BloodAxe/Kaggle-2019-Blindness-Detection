#!/usr/bin/env bash

python train_regression_baseline.py -m reg_resnext50_rms -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -es 20 -v --use-messidor --warmup 2
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -es 20 -v --use-idrid --warmup 2
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 50 -es 10 -v --use-aptos2019 --warmup 1
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -es 20 -v --use-aptos2015 --warmup 2
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a medium -f 0 -b 60 --fp16 -o Adam -lr 3e-4 --balance -s multistep -e 100 -es 20 -v --use-aptos2019 --use-idrid --use-messidor --warmup 2
sleep 5

python train_regression_baseline.py -m reg_efficientb4_rms -l wing_loss -a medium -f 0 -b 36 --fp16 -o Adam --balance -s multistep -lr 3e-4 -e 100 -es 20 -v --use-aptos1029 --use-idrid --use-messidor
sleep 5