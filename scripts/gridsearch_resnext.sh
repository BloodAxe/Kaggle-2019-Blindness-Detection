#!/usr/bin/env bash


python train_reg.py -m seresnext50d_gap\
    -a medium -f 0 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50\
    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4\
    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth\
    -p redfree
sleep 15

#python train_reg.py -m seresnext50d_gap\
#    -a medium -f 0 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50\
#    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4\
#    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth\
#    -p clahe
#sleep 15

python train_reg.py -m seresnext50d_gap\
    -a medium -f 0 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50\
    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4\
    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth\
    -p unsharp
sleep 15

#python train_reg.py -m seresnext50d_gap\
#    -a medium -f 0 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50\
#    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4\
#    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth\
#    -d 0.5
#sleep 15

python train_reg.py -m seresnext50d_gap\
    -a medium -f 0 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50\
    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4\
    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth\
    --criterion-reg wing_loss

sleep 15

#python train_reg.py -m seresnext50d_gap\
#    -a medium -f 0 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50\
#    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-3\
#    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth
#

python train_reg.py -m seresnext50d_gap\
    -a medium -f 0 -b 18 --size 512 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 100\
    --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 1e-4\
    -t seresnext50d_gap_512_medium_aptos2019_messidor_idrid_fold0_admiring_wright.pth\
    --unsupervised
sleep 15
