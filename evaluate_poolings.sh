#!/usr/bin/env bash

#python train_classifier_baseline.py -m cls_resnext50_gap    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
#sleep 5

python train_classifier_baseline.py -m cls_resnext50_gmp    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_classifier_baseline.py -m cls_resnext50_gwap   -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

#python train_classifier_baseline.py -m cls_resnext50_ocp    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
#sleep 5

python train_classifier_baseline.py -m cls_resnext50_rms    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_classifier_baseline.py -m cls_resnext50_maxavg -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5


python train_regression_baseline.py -m reg_resnext50_gap    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_regression_baseline.py -m reg_resnext50_gmp    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_regression_baseline.py -m reg_resnext50_gwap   -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_regression_baseline.py -m reg_resnext50_ocp    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms    -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5

python train_regression_baseline.py -m reg_resnext50_maxavg -b 60 -e 50 -es 20 -lr 1e-4 -o Adam -a hard --fast --fp16
sleep 5
