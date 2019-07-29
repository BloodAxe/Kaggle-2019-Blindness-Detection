#!/usr/bin/env bash

python train_regression_baseline.py -m reg_seresnext50_avg -a hard -f 0 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-5 -e 20 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_seresnext50_rms_512_fold0_hard_keen_williams_aptos2019_messidor_idrid_best.pth --use-aptos2015 --unsupervised
sleep 5

python train_regression_baseline.py -m reg_seresnext50_avg -a hard -f 1 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-5 -e 20 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_seresnext50_rms_512_fold1_hard_sleepy_pare_aptos2019_messidor_idrid_best.pth --use-aptos2015 --unsupervised
sleep 5

python train_regression_baseline.py -m reg_seresnext50_avg -a hard -f 2 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-5 -e 20 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_seresnext50_rms_512_fold2_hard_jolly_edison_aptos2019_messidor_idrid_best.pth --use-aptos2015 --unsupervised
sleep 5

python train_regression_baseline.py -m reg_seresnext50_avg -a hard -f 3 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-5 -e 20 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer reg_seresnext50_rms_512_fold3_hard_zealous_agnesi_aptos2019_messidor_idrid_best.pth --use-aptos2015 --unsupervised

python train_regression_baseline.py -m reg_seresnext50_avg -a hard -f 0 -b 60 --fp16 -o Adam -d 0.5\
    -s multistep -lr 1e-5 -e 20 -v --use-idrid --use-messidor --use-aptos2019\
    --transfer runs/reg/reg_seresnext50_avg/Jul27_16_53/reg_seresnext50_avg_512_fold0_hard_laughing_kepler_aptos2019_messidor_idrid/checkpoints/best.pth --use-aptos2015 --unsupervised