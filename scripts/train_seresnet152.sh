#!/usr/bin/env bash

# Pretrain model on past data
#python train_ord_universal.py -m seresnet152_gapv2 \
#    -a medium -b 42 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
#    -e 5 \
#    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
#    --valid-on aptos-2019-train idrid-train idrid-test messidor\
#    -v --criterion-ord mse -x seresnet152_gapv2_pretrain
#sleep 15

# Train 4 folds on this data
python train_ord.py -m seresnet152_gap \
    -a medium -d 0.5 -b 42 --size 512 --fp16 -o Ranger -wd 1e-3 -s simple -lr 3e-5\
    --warmup 10 \
    --epochs 75 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 --seed 104 \
    -v --criterion-ord huber -t seresnet152_gapv2_pretrain.pth
sleep 15

python train_ord.py -m seresnet152_gap \
    -a medium -d 0.5 -b 42 --size 512 --fp16 -o Ranger -wd 1e-3 -s simple -lr 3e-5\
    --warmup 10 \
    --epochs 75 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 1 --seed 105 \
    -v --criterion-ord huber -t seresnet152_gapv2_pretrain.pth
sleep 15

python train_ord.py -m seresnet152_gap \
    -a medium -d 0.5 -b 42 --size 512 --fp16 -o Ranger -wd 1e-3 -s simple -lr 3e-5\
    --warmup 10 \
    --epochs 75 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 2 --seed 106 \
    -v --criterion-ord huber -t seresnet152_gapv2_pretrain.pth
sleep 15

python train_ord.py -m seresnet152_gap \
    -a medium -d 0.5 -b 42 --size 512 --fp16 -o Ranger -wd 1e-3 -s simple -lr 3e-5\
    --warmup 10 \
    --epochs 75 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 3 --seed 107 \
    -v --criterion-ord huber -t seresnet152_gapv2_pretrain.pth
sleep 15