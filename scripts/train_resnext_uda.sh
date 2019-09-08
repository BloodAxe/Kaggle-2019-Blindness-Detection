#!/usr/bin/env bash

python train_cls_uda.py -m seresnext50d_gap -a medium -f 0 -b 18 --size 512 -o AdamW -wd 1e-4\
  -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 --use-aptos2015 -v --criterion-reg clipped_mse --criterion-ord mse

python train_cls_uda.py -m seresnext50d_gwap -a medium -f 0 -b 18 --size 512 -o AdamW -wd 1e-4\
  -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 --use-aptos2015 -v --criterion-reg clipped_mse --criterion-ord mse

python train_cls_uda.py -m resnet50_gap -a medium -f 0 -b 30 --size 512 -o SGD -wd 1e-4\
  -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 --use-aptos2015 -v --criterion-reg clipped_mse --criterion-ord mse

python train_cls_uda.py -m efficientb5_gwap -a medium -f 0 -b 18 --size 512 -o lamb -wd 1e-4\
  -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 --use-aptos2015 -v --criterion-reg clipped_mse --criterion-ord mse
