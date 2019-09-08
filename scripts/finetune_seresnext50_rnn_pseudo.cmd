
REM python train_ord.py -m seresnext50_rnn^
REM     -a medium -b 72 -w 12 --size 512 -o RAdam -wd 1e-4 -s simple -lr 3e-4^
REM     --warmup 25 -e 0 --fine-tune 10 ^
REM     -f 0 ^
REM     --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 ^
REM     -v -d 0.5 --criterion-ord huber ^
REM     --seed 80 ^
REM     -c models/Aug31_09_20_seresnext50_rnn_clever_roentgen/seresnext50_rnn_512_hard_aptos2019_messidor_idrid_fold0_clever_roentgen.pth


REM python train_ord.py -m seresnext50_rnn^
REM     -a medium -b 72 -w 12 --size 512 -o RAdam -wd 1e-4 -s simple -lr 3e-4^
REM     --warmup 25 -e 0 --fine-tune 10 ^
REM     -f 1 ^
REM     --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 ^
REM     -v -d 0.5 --criterion-ord huber ^
REM     --seed 81 ^
REM     -c models/Aug31_09_20_seresnext50_rnn_clever_roentgen/seresnext50_rnn_512_hard_aptos2019_messidor_idrid_fold1_clever_roentgen.pth


REM python train_ord.py -m seresnext50_rnn^
REM     -a medium -b 72 -w 12 --size 512 -o RAdam -wd 1e-4 -s simple -lr 3e-4^
REM     --warmup 25 -e 0 --fine-tune 10 ^
REM     -f 2 ^
REM     --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 ^
REM     -v -d 0.5 --criterion-ord huber ^
REM     --seed 82 ^
REM     -c models/Aug31_09_20_seresnext50_rnn_clever_roentgen/seresnext50_rnn_512_hard_aptos2019_messidor_idrid_fold2_clever_roentgen.pth


REM python train_ord.py -m seresnext50_rnn^
REM     -a medium -b 72 -w 12 --size 512 -o RAdam -wd 1e-4 -s simple -lr 3e-4^
REM     --warmup 25 -e 0 --fine-tune 10 ^
REM     -f 3 ^
REM     --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 ^
REM     -v -d 0.5 --criterion-ord huber ^
REM     --seed 83 ^
REM     -c models/Aug31_09_20_seresnext50_rnn_clever_roentgen/seresnext50_rnn_512_hard_aptos2019_messidor_idrid_fold3_clever_roentgen.pth


python train_ord.py -m seresnext50_rnn^
    -a medium -b 72 -w 12 --size 512 -o RAdam -wd 1e-4 -s simple -lr 3e-4^
    --warmup 25 -e 0 --fine-tune 10 ^
    -f 2 ^
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 ^
    -v -d 0.5 --criterion-ord huber ^
    --seed 82 ^
    -c runs/Sep06_10_50/seresnext50_rnn_512_medium_aptos2019_messidor_idrid_pl1_fold2_elastic_babbage/warmup/checkpoints/best.pth


python train_ord.py -m seresnext50_rnn^
    -a medium -b 72 -w 12 --size 512 -o RAdam -wd 1e-4 -s simple -lr 3e-4^
    --warmup 25 -e 0 --fine-tune 10 ^
    -f 3 ^
    --use-aptos2019 --use-aptos2019-test-pl1 --use-idrid --use-messidor --use-messidor2-pl1 --use-aptos2015-pl1 ^
    -v -d 0.5 --criterion-ord huber ^
    --seed 83 ^
    -c runs/Sep06_15_23/seresnext50_rnn_512_medium_aptos2019_messidor_idrid_pl1_fold3_vigilant_carson/warmup/checkpoints/last.pth

