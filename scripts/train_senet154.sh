#!/usr/bin/env bash

# Pretrain model on past data
python train_ord_universal.py senet154_gapv2\
    -a medium -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 1 \
    -e 20 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid messidor\
    -v --criterion-ord mse -x pnasnet5_gap_pretrain
sleep 15

# Train 4 folds on this data
python train_ord.py -m senet154_gapv2 \
    -a medium -d 0.5 -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-4 \
    --epochs 100 \
    --fine-tune 25 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 --seed 100 \
    -v --criterion-ord huber -t pnasnet5_gap_pretrain.pth
sleep 15

python train_ord.py -m senet154_gapv2 \
    -a medium -d 0.5 -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-4 \
    --epochs 100 \
    --fine-tune 25 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 1 --seed 101 \
    -v --criterion-ord huber -t pnasnet5_gap_pretrain.pth
sleep 15

python train_ord.py -m senet154_gapv2 \
    -a medium -d 0.5 -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-4 \
    --epochs 100 \
    --fine-tune 25 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 2 --seed 102 \
    -v --criterion-ord huber -t pnasnet5_gap_pretrain.pth
sleep 15

python train_ord.py -m senet154_gapv2 \
    -a medium -d 0.5 -b 18 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-4 \
    --epochs 100 \
    --fine-tune 25 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 3 --seed 103 \
    -v --criterion-ord huber -t pnasnet5_gap_pretrain.pth
sleep 15