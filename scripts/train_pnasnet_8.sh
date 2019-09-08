#!/usr/bin/env bash

# Pretrain model on past data
python train_ord_universal.py -m pnasnet5_gapv2\
    -a medium -b 32  -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -lr 3e-4\
    -e 5 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid-train idrid-test messidor\
    -v --criterion-ord mse -x pnasnet5_gapv2_pretrain
sleep 15

# Train 4 folds on this data
python train_ord.py -m pnasnet5_gapv2 \
    -a medium -d 0.5 -b 32  -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    --epochs 100 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 --seed 100 \
    -v --criterion-ord huber -t pnasnet5_gapv2_pretrain.pth
sleep 15

python train_ord.py -m pnasnet5_gapv2 \
    -a medium -d 0.5 -b 32  -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    --epochs 100 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 1 --seed 101 \
    -v --criterion-ord huber -t pnasnet5_gapv2_pretrain.pth
sleep 15

python train_ord.py -m pnasnet5_gapv2 \
    -a medium -d 0.5 -b 32  -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    --epochs 100 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 2 --seed 102 \
    -v --criterion-ord huber -t pnasnet5_gapv2_pretrain.pth
sleep 15

python train_ord.py -m pnasnet5_gapv2 \
    -a medium -d 0.5 -b 32  -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    --epochs 100 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 3 --seed 103 \
    -v --criterion-ord huber -t pnasnet5_gapv2_pretrain.pth
sleep 15