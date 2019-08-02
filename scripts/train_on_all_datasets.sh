#!/usr/bin/env bash

#python train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
#    -s multistep -lr 1e-4 -e 100 -es 20 -v --use-idrid --warmup 10
#sleep 5
#
#python train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
#    -s multistep -lr 1e-4 -e 100 -es 20 -v --use-messidor --warmup 10
#sleep 5
#
#python train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
#    -s multistep -lr 1e-4 -e 100 -es 20 -v --use-aptos2019 --warmup 10
#sleep 5
#
#python train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
#    -s multistep -lr 1e-4 -e 100 -es 20 -v --use-aptos2019 --use-messidor --use-idrid --warmup 10
#sleep 5

python train_regression_baseline.py -m reg_seresnext50_rms -a hard -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-aptos2015
sleep 5
