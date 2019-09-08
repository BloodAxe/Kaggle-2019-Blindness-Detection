#!/usr/bin/env bash

# Pretrain model on past data
python train_ord_universal.py -m resnet152_gap\
    -a medium -b 96 -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    -e 10 \
    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
    --valid-on aptos-2019-train idrid-test idrid-train messidor\
    -v --criterion-ord mse -x resnet152_gap_pretrain
sleep 15

# Train 4 folds on this data
python train_ord.py -m resnet152_gap \
    -a medium -d 0.5 -b 96 -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-5 \
    --epochs 50 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 --seed 100 \
    -v --criterion-ord huber -t resnet152_gap_pretrain.pth
sleep 15

python train_ord.py -m resnet152_gap \
    -a medium -d 0.5 -b 96 -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-5 \
    --epochs 50 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 1 --seed 101 \
    -v --criterion-ord huber -t resnet152_gap_pretrain.pth
sleep 15

python train_ord.py -m resnet152_gap \
    -a medium -d 0.5 -b 96 -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-5 \
    --epochs 50 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 2 --seed 102 \
    -v --criterion-ord huber -t resnet152_gap_pretrain.pth
sleep 15

python train_ord.py -m resnet152_gap \
    -a medium -d 0.5 -b 96 -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-5 \
    --epochs 50 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 3 --seed 103 \
    -v --criterion-ord huber -t resnet152_gap_pretrain.pth
sleep 15