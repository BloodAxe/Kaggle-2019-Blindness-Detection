#!/usr/bin/env bash

# Pretrain model to start with something more or less accurate
#python train_classifier_baseline.py -m cls_seresnext50_max -a light -f 0 -b 60 --fp16 -o Adam -lr 1e-4\
#  -e 20 -v --use-aptos2019 --use-idrid --use-messidor --warmup 20 --criterion ce -x cls_seresnext50_max_pretrain
#sleep 5


# Let's check different loss functions
# Loss function: soft_ce
#python train_classifier_baseline.py -m cls_seresnext50_max -a light -f 0 -b 60 --fp16 -o Adam -d 0.5 -lr 1e-4\
#  -e 50 -v --use-aptos2019 --use-idrid --use-messidor --criterion soft_ce --transfer cls_seresnext50_max_pretrain.pth
#sleep 5

# Let's check different loss functions
# Loss function: hybrid_kappa
#python train_classifier_baseline.py -m cls_seresnext50_max -a light -f 0 -b 60 --fp16 -o Adam -d 0.5 -lr 1e-4\
#  -e 50 -v --use-aptos2019 --use-idrid --use-messidor --criterion hybrid_kappa --transfer cls_seresnext50_max_pretrain.pth
#sleep 5

# Let's check different loss functions
# Loss function: focal
python train_classifier_baseline.py -m cls_seresnext50_max -a light -f 0 -b 48 --fp16 -o Adam -d 0.5 -lr 1e-4\
  -e 50 -v --use-aptos2019 --use-idrid --use-messidor --criterion focal --transfer cls_seresnext50_max_pretrain.pth
sleep 5



# Let's check different augmentations
#python train_classifier_baseline.py -m cls_seresnext50_max -a medium -f 0 -b 60 --fp16 -o Adam -d 0.5 -lr 1e-4\
#  -e 50 -v --use-aptos2019 --use-idrid --use-messidor --transfer cls_seresnext50_max_pretrain.pth
#sleep 5
#
#python train_classifier_baseline.py -m cls_seresnext50_max -a hard -f 0 -b 60 --fp16 -o Adam -d 0.5 -lr 1e-4\
#  -e 50 -v --use-aptos2019 --use-idrid --use-messidor --transfer cls_seresnext50_max_pretrain.pth
#sleep 5
#
## Let's check different preprocessing
#python train_classifier_baseline.py -m cls_seresnext50_max -a light -f 0 -b 60 --fp16 -o Adam -d 0.5 -lr 1e-4\
#  -e 50 -v --use-aptos2019 --use-idrid --use-messidor -p unsharp --transfer cls_seresnext50_max_pretrain.pth
#sleep 5
#
#python train_classifier_baseline.py -m cls_seresnext50_max -a light -f 0 -b 60 --fp16 -o Adam -d 0.5 -lr 1e-4\
#  -e 50 -v --use-aptos2019 --use-idrid --use-messidor -p clahe --transfer cls_seresnext50_max_pretrain.pth
#sleep 5
