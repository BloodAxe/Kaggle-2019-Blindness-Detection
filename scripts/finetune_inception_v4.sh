#!/usr/bin/env bash


#python train_ord_universal.py -m inceptionv4_gap\
#    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    -e 50 \
#    --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public\
#    --valid-on aptos-2019-train idrid-train idrid-test messidor\
#    -v --criterion-cls soft_ce --criterion-ord huber -x inceptionv4_gap_pretrain
#sleep 15

#    -f 0 -f 1 -f 2 -f 3 \
#python train_ord.py -m inceptionv4_gap\
#    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    --epochs 100 --fine-tune 25 \
#    -f 3 \
#    --use-idrid --use-messidor --use-aptos2019 \
#    -v -d 0.5 -wd 1e-4 --criterion-ord huber -t inceptionv4_gap_pretrain.pth \
#    --seed 22
#
#sleep 15

#python train_ord.py -m inceptionv4_gap\
#    -a medium -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    --epochs 100 --fine-tune 25 \
#    -f 2 \
#    --use-idrid --use-messidor --use-aptos2019 \
#    -v -d 0.5 -wd 1e-4 --criterion-ord huber -t inceptionv4_gap_pretrain.pth \
#    --seed 22
#
#sleep 15

#python train_ord.py -m inceptionv4_gwap\
#    -a hard -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
#    --warmup 10 --epochs 50 \
#    -f 0 -f 1 -f 2 -f 3 \
#    --use-idrid --use-messidor --use-aptos2019 \
#    -v -d 0.15 -wd 1e-4 --criterion-ord mse -t inceptionv4_gap_pretrain.pth \
#    --seed 23

python train_ord.py -m inceptionv4_gwap\
    -a hard -b 96 -w 16 --size 512 --fp16 -o RAdam -wd 1e-4 -s simple -lr 3e-4\
    --warmup 10 --epochs 50 \
    -f 2 -f 3 \
    --use-idrid --use-messidor --use-aptos2019 \
    -v -d 0.15 -wd 1e-4 --criterion-ord mse -t inceptionv4_gap_pretrain.pth \
    --seed 24
