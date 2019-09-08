REM #!/usr/bin/env bash

REM # Pretrain model on past data
REM python train_ord_universal.py -m resnet34_gap ^
REM     -a medium -b 32 --size 512 -o Ranger -wd 1e-5 -s simple -lr 3e-4^
REM     --warmup 1 ^
REM     -e 20 ^
REM     --train-on aptos-2015-train aptos-2015-test-private aptos-2015-test-public^
REM     --valid-on aptos-2019-train idrid-test idrid-train messidor^
REM     -v --criterion-ord mse -x resnet34_gap_pretrain

REM # Train 4 folds on this data
python train_ord.py -m resnet34_gap ^
    -a medium -d 0.5 -b 32 --size 512 -o Ranger -wd 1e-5 -s simple -lr 3e-4^
    --epochs 100 ^
    --fine-tune 25 ^
    --use-aptos2019 --use-idrid --use-messidor ^
    -f 0 --seed 100 ^
    -v --criterion-ord huber -t resnet34_gap_pretrain.pth

python train_ord.py -m resnet34_gap ^
    -a medium -d 0.5 -b 32 --size 512 -o Ranger -wd 1e-5 -s simple -lr 3e-4^
    --epochs 100 ^
    --fine-tune 25 ^
    --use-aptos2019 --use-idrid --use-messidor ^
    -f 1 --seed 101 ^
    -v --criterion-ord huber -t resnet34_gap_pretrain.pth

python train_ord.py -m resnet34_gap ^
    -a medium -d 0.5 -b 32 --size 512 -o Ranger -wd 1e-5 -s simple -lr 3e-4^
    --epochs 100 ^
    --fine-tune 25 ^
    --use-aptos2019 --use-idrid --use-messidor ^
    -f 2 --seed 102 ^
    -v --criterion-ord huber -t resnet34_gap_pretrain.pth

python train_ord.py -m resnet34_gap ^
    -a medium -d 0.5 -b 32 --size 512 -o Ranger -wd 1e-5 -s simple -lr 3e-4^
    --epochs 100 ^
    --fine-tune 25 ^
    --use-aptos2019 --use-idrid --use-messidor ^
    -f 3 --seed 103 ^
    -v --criterion-ord huber -t resnet34_gap_pretrain.pth
