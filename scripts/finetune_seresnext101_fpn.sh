#!/usr/bin/env bash

python train_ord.py -m seresnext101_fpn \
    -a hard -d 0.5 -b 48 --fp16 --size 512 -o RMS -wd 1e-3 -s simple -lr 5e-4\
    --warmup 25 \
    --epochs 50 \
    --fine-tune 10 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 0 --seed 10 \
    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold0_modest_williams.pth
sleep 15

python train_ord.py -m seresnext101_fpn \
    -a hard -d 0.5 -b 48 --fp16 --size 512 -o RMS -wd 1e-3 -s simple -lr 5e-4\
    --warmup 25 \
    --epochs 50 \
    --fine-tune 10 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 1 --seed 11 \
    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold1_modest_williams.pth
sleep 15

python train_ord.py -m seresnext101_fpn \
    -a hard -d 0.5 -b 48 --fp16 --size 512 -o RMS -wd 1e-3 -s simple -lr 5e-4\
    --warmup 25 \
    --epochs 50 \
    --fine-tune 10 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 2 --seed 12 \
    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold2_modest_williams.pth
sleep 15

python train_ord.py -m seresnext101_fpn \
    -a hard -d 0.5 -b 48 --fp16 --size 512 -o RMS -wd 1e-3 -s simple -lr 5e-4\
    --warmup 25 \
    --epochs 50 \
    --fine-tune 10 \
    --use-aptos2019 --use-idrid --use-messidor \
    -f 3 --seed 13 \
    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold3_modest_williams.pth
sleep 15
