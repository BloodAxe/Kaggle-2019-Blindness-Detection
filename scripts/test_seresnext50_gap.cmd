REM FineTune


REM Train on IDRID and Messidor sets, validate on all Aptos2019
REM python train_ord_universal.py -m seresnext50_gap -a medium -w 4 -b 8 -o RAdam -wd 1e-5 -s simple -lr 3e-4 -v -l1 1e-5^
REM     --warmup 10 --epochs 50 ^
REM     --train idrid-train idrid-test messidor ^
REM     --valid aptos-2019-train

REM Train on Aptos2019, validate on IDRID and Messidor
REM python train_ord_universal.py -m seresnext50_gap -a medium -w 4 -b 8 -o RAdam -wd 1e-5 -s simple -lr 3e-4 -v -l1 1e-5^
REM     --warmup 10 --epochs 50 ^
REM     --train aptos-2019-train^
REM     --valid idrid-train idrid-test messidor

REM Train on Aptos2019, validate on IDRID and Messidor
python train_ord_universal.py -m seresnext50_gap -a medium -w 4 -b 8 -o RAdam -wd 1e-5 -s simple -lr 3e-4 -v -l1 1e-5^
    --warmup 10 --epochs 50 ^
    --train aptos-2015-train^
    --valid aptos-2019-train
