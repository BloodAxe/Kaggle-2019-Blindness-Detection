#!/usr/bin/env bash

python train_regression_baseline.py -m reg_resnext50_rms --size 768 --fp16 -b 30 -e 100 -l wing_loss -lr 1e-4 -wd 1e-3 -o Adam -a hard -f 0 --use-aptos2019 --use-idrid --use-messidor -c pretrained/cls_resnext50_aptos2019_idrid_messidor/reg_resnext50_rms_eager_pike_fold0_aptos2019_messidor_idrid_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms --size 768 --fp16 -b 30 -e 100 -l wing_loss -lr 1e-4 -wd 1e-3 -o Adam -a hard -f 1 --use-aptos2019 --use-idrid --use-messidor -c pretrained/cls_resnext50_aptos2019_idrid_messidor/reg_resnext50_rms_512_affectionate_mayer_fold1_aptos2019_messidor_idrid_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms --size 768 --fp16 -b 30 -e 100 -l wing_loss -lr 1e-4 -wd 1e-3 -o Adam -a hard -f 2 --use-aptos2019 --use-idrid --use-messidor -c pretrained/cls_resnext50_aptos2019_idrid_messidor/reg_resnext50_rms_512_kind_wing_fold2_aptos2019_messidor_idrid_best.pth
sleep 5

python train_regression_baseline.py -m reg_resnext50_rms --size 768 --fp16 -b 30 -e 100 -l wing_loss -lr 1e-4 -wd 1e-3 -o Adam -a hard -f 3 --use-aptos2019 --use-idrid --use-messidor -c pretrained/cls_resnext50_aptos2019_idrid_messidor/reg_resnext50_rms_512_thirsty_noyce_fold3_aptos2019_messidor_idrid_best.pth
sleep 5
