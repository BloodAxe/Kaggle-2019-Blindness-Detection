#!/usr/bin/env bash


#python train_cls.py -m seresnext50d_gap -a medium -f 0 -b 60 --fp16 --size 512 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 100 -w 8 --use-idrid --use-aptos2019 -v -d 0.25 --warmup 15
#python train_reg.py -m seresnext50d_gap -a medium -f 0 -b 60 --fp16 --size 512 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 100 -w 8 --use-idrid --use-aptos2019 -v -d 0.25 --warmup 15
#python train_ord.py -m seresnext50d_gap -a medium -f 0 -b 60 --fp16 --size 512 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 100 -w 8 --use-idrid --use-aptos2019 -v -d 0.25 --warmup 15

python train_ord.py -m seresnext50d_gwap -a hard -f 0 -f 1 -f 2 -f 3 -b 30 --fp16 --size 768 -o AdamW -wd 1e-4 -s multistep -lr 5e-4 -e 50 --use-idrid --use-aptos2019 -v -d 0.25\
    --criterion-cls focal_kappa 0.25 --criterion-reg mse 0.25 --criterion-ord mse 1.0