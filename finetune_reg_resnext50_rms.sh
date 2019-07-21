#!/usr/bin/env bash

python train_regression_baseline.py -m reg_resnext50_rms -b 60 -e 100 -es 20 -l wing_loss -lr 1e-4 -o Adam -a hard --fp16 -f 0 --use-aptos2019 --use-idrid --use-messidor -c pretrained/791/reg_resnext50_rms/fold_0/Jul18_15_49_wing_loss_fp16_fast/checkpoints/reg_resnext50_rms_fold0_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -b 60 -e 100 -es 20 -l wing_loss -lr 1e-4 -o Adam -a hard --fp16 -f 1 --use-aptos2019 --use-idrid --use-messidor -c pretrained/791/reg_resnext50_rms/fold_1/Jul18_17_17_wing_loss_fp16_fast/checkpoints/reg_resnext50_rms_fold1_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -b 60 -e 100 -es 20 -l wing_loss -lr 1e-4 -o Adam -a hard --fp16 -f 2 --use-aptos2019 --use-idrid --use-messidor -c pretrained/791/reg_resnext50_rms/fold_2/Jul18_18_45_wing_loss_fp16_fast/checkpoints/reg_resnext50_rms_fold2_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -b 60 -e 100 -es 20 -l wing_loss -lr 1e-4 -o Adam -a hard --fp16 -f 2 --use-aptos2019 --use-idrid --use-messidor -c pretrained/791/reg_resnext50_rms/fold_3/Jul18_20_14_wing_loss_fp16_fast/checkpoints/reg_resnext50_rms_fold3_best.pth
sleep 5
