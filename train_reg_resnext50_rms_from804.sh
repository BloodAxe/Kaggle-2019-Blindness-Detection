#!/usr/bin/env bash

python train_regression_baseline.py -m reg_resnext50_rms -a hard -f 0 -b 60 -l huber --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_resnext50_rms_eager_pike_fold0_aptos2019_messidor_idrid_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a hard -f 1 -b 60 -l huber --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_resnext50_rms_512_affectionate_mayer_fold1_aptos2019_messidor_idrid_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a hard -f 2 -b 60 -l huber --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_resnext50_rms_512_kind_wing_fold2_aptos2019_messidor_idrid_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms -a hard -f 3 -b 60 -l huber --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-4 -e 100 -es 10 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_resnext50_rms_512_thirsty_noyce_fold3_aptos2019_messidor_idrid_best.pth
