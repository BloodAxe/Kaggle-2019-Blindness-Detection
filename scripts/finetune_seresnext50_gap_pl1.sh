#!/usr/bin/env bash

python train_ord.py -m seresnext50_gap \
    -a medium -d 0.5 -b 60 --fp16 --size 512 -o Ranger -wd 1e-4 -s simple -lr 3e-5\
    --epochs 15 \
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 --use-aptos2015-test-pl1\
    -f 0 --seed 100 \
    -v --criterion-ord huber -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth
sleep 15

python train_ord.py -m seresnext50_gap \
    -a medium -d 0.5 -b 60 --fp16 --size 512 -o Ranger -wd 1e-4 -s simple -lr 3e-5\
    --epochs 15 \
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 --use-aptos2015-test-pl1\
    -f 1 --seed 100 \
    -v --criterion-ord huber -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold1_heuristic_sinoussi.pth
sleep 15

python train_ord.py -m seresnext50_gap \
    -a medium -d 0.5 -b 60 --fp16 --size 512 -o Ranger -wd 1e-4 -s simple -lr 3e-5\
    --epochs 15 \
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 --use-aptos2015-test-pl1\
    -f 2 --seed 100 \
    -v --criterion-ord huber -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold2_heuristic_sinoussi.pth
sleep 15

python train_ord.py -m seresnext50_gap \
    -a medium -d 0.5 -b 60 --fp16 --size 512 -o Ranger -wd 1e-4 -s simple -lr 3e-5\
    --epochs 15 \
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 --use-aptos2015-test-pl1\
    -f 3 --seed 100 \
    -v --criterion-ord huber -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold3_heuristic_sinoussi.pth
sleep 15
