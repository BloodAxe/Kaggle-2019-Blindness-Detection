#!/usr/bin/env bash

# Pretrain model on past data
python train_ord_universal.py -m densenet201_gap\
    -a medium -b 36 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -e 50 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-cls focal_kappa --criterion-ord huber -x densenet201_gap_pretrain
sleep 15

# Train 4 folds on this data
python train_ord.py -m densenet201_gwap \
    -a medium -d 0.5 -b 36 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-4 \
    --warmup 5 \
    --epochs 150 \
    --fine-tune 25 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 -f 1 -f 2 -f 3 \
    -v --criterion-ord huber -t densenet201_gap_pretrain.pth