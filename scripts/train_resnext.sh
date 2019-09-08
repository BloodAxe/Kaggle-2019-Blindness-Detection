#!/usr/bin/env bash

# Pretrain models
python train_ord_universal.py -m seresnext50d_gap\
    -a medium -b 60 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -e 50 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-cls focal_kappa --criterion-ord huber --criterion-reg huber -x seresnext50d_gap_pretrain
sleep 15

python train_ord_universal.py -m resnet34_gap\
    -a medium -b 72 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -e 50 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-cls focal_kappa --criterion-ord huber --criterion-reg huber -x resnet34_gap_pretrain
sleep 15

python train_ord_universal.py -m seresnext101_gap\
    -a medium -b 48 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -e 50 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-cls focal_kappa --criterion-ord huber --criterion-reg huber -x seresnext101_gap_pretrain
sleep 15

python train_ord_universal.py -m densenet201_gap\
    -a medium -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -e 50 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-cls focal_kappa --criterion-ord huber --criterion-reg huber -x densenet201_gap_pretrain
sleep 15

python train_ord_universal.py -m efficientb4_gap\
    -a medium -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -e 50 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-cls focal_kappa --criterion-ord huber --criterion-reg huber -x efficientb4_gap_pretrain
sleep 15

#python train_ord.py -m seresnext50d_gap\
#    -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    -e 50 \
#    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa 0.1 --criterion-ord mse
#sleep 15
#
#python train_ord.py -m seresnext50d_gapv2\
#    -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    -e 50 \
#    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa 0.1 --criterion-ord mse
#sleep 15
#
#python train_ord.py -m resnet34_max\
#    -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    -e 50 \
#    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls soft_ce 0.1 --criterion-ord huber
#sleep 15
#
