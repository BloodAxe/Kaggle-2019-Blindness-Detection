#!/usr/bin/env bash

# Train 4 folds on this data
python train_ord.py -m pnasnet5_gap \
    -a hard -d 0.5 -b 32 -w 16 --size 512 --fp16 -o Ranger -wd 1e-4 -s simple -lr 3e-4\
    -l1 1e-5\
    --epochs 100 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 -f 1 -f 2 -f 3 --seed 100 \
    -v --criterion-ord huber