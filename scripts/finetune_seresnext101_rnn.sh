#!/usr/bin/env bash

#python train_ord.py -m seresnext101_rnn \
#    -a hard2 -d 0.5 -b 48 --fp16 --size 512 -o RAdam -wd 1e-4 -s multistep -lr 4e-3\
#    --warmup 25 \
#    --epochs 100 \
#    --fine-tune 50 \
#    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1\
#    -f 0 --seed 10 \
#    -v --criterion-ord mse -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold0_modest_williams.pth
#sleep 15

python train_ord.py -m seresnext101_rnn \
    -a hard2 -d 0.5 -b 48 --fp16 --size 512 -o RAdam -wd 1e-4 -s multistep -lr 4e-3\
    --epochs 100 \
    --fine-tune 50 \
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1\
    -f 0 --seed 10 \
    -v --criterion-ord mse -c runs/Sep05_10_35/seresnext101_rnn_512_hard2_aptos2019_messidor_idrid_pl1_fold0_admiring_boyd/warmup/checkpoints/best.pth
sleep 15

#python train_ord.py -m seresnext101_rnn \
#    -a hard2 -d 0.5 -b 48 --fp16 --size 512 -o SGD -wd 4e-4 -s multistep -lr 5e-4\
#    --warmup 100 \
#    --epochs 50 \
#    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1\
#    -f 1 --seed 11 \
#    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold1_modest_williams.pth
#sleep 15
#
#python train_ord.py -m seresnext101_rnn \
#    -a hard2 -d 0.5 -b 48 --fp16 --size 512 -o SGD -wd 4e-4 -s multistep -lr 5e-4\
#    --warmup 100 \
#    --epochs 50 \
#    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 \
#    -f 2 --seed 12 \
#    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold2_modest_williams.pth
#sleep 15
#
#python train_ord.py -m seresnext101_rnn \
#    -a hard2 -d 0.5 -b 48 --fp16 --size 512 -o SGD -wd 4e-4 -s multistep -lr 5e-4\
#    --warmup 100 \
#    --epochs 50 \
#    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1\
#    -f 3 --seed 13 \
#    -v --criterion-ord cauchy -t models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold3_modest_williams.pth
#sleep 15
