#!/usr/bin/env bash

python train_ord.py -m inceptionv4_gap\
    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 0 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 72 \
    -c models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold0_happy_wright.pth

sleep 15

python train_ord.py -m inceptionv4_gap\
    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 1 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 73 \
    -c models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold1_happy_wright.pth

sleep 15

python train_ord.py -m inceptionv4_gap\
    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 2 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 74 \
    -c models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold2_happy_wright.pth

sleep 15

python train_ord.py -m inceptionv4_gap\
    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 3 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 75 \
    -c models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold3_happy_wright.pth

sleep 15

# FineTune SeResNext101

python train_ord.py -m seresnext101_gap\
    -a medium -b 48 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 0 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 76 \
    -c models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold0_modest_williams.pth

sleep 15

python train_ord.py -m seresnext101_gap\
    -a medium -b 48 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 1 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 77 \
    -c models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold1_modest_williams.pth

sleep 15

python train_ord.py -m seresnext101_gap\
    -a medium -b 48 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 2 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 78 \
    -c models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold2_modest_williams.pth

sleep 15

python train_ord.py -m seresnext101_gap\
    -a medium -b 48 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 3 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 79 \
    -c models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold3_modest_williams.pth

sleep 15

# FineTune SeResNext50

python train_ord.py -m seresnext50_gap\
    -a medium -b 60 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 0 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 80 \
    -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth

sleep 15

python train_ord.py -m seresnext50_gap\
    -a medium -b 60 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 1 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 81 \
    -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold1_heuristic_sinoussi.pth

sleep 15

python train_ord.py -m seresnext50_gap\
    -a medium -b 60 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 2 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 82 \
    -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold2_heuristic_sinoussi.pth

sleep 15

python train_ord.py -m seresnext50_gap\
    -a medium -b 60 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 25 --epochs 5 \
    -f 3 \
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 \
    -v -d 0.5 --criterion-ord huber \
    --seed 83 \
    -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold3_heuristic_sinoussi.pth

sleep 15