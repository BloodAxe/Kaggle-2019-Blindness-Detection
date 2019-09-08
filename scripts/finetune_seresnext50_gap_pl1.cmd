REM python train_ord.py -m seresnext50_gap ^
REM     -a medium -d 0.5 -b 12 --size 512 -o SGD -wd 1e-4 -s simple -lr 3e-5^
REM     --warmup 15 ^
REM     --epochs 10 ^
REM     --fine-tune 20 ^
REM     --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 ^
REM     -f 0 --seed 100 ^
REM     -v --criterion-ord cauchy -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth

python train_ord.py -m seresnext50_gap ^
    -a medium -d 0.5 -b 12 --size 512 -o SGD -wd 1e-4 -s simple -lr 3e-5^
    --warmup 15 ^
    --epochs 10 ^
    --fine-tune 20 ^
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 ^
    -f 1 --seed 100 ^
    -v --criterion-ord cauchy -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold1_heuristic_sinoussi.pth

python train_ord.py -m seresnext50_gap ^
    -a medium -d 0.5 -b 12 --size 512 -o SGD -wd 1e-4 -s simple -lr 3e-5^
    --warmup 15 ^
    --epochs 10 ^
    --fine-tune 20 ^
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 ^
    -f 2 --seed 100 ^
    -v --criterion-ord cauchy -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold2_heuristic_sinoussi.pth

python train_ord.py -m seresnext50_gap ^
    -a medium -d 0.5 -b 12 --size 512 -o SGD -wd 1e-4 -s simple -lr 3e-5^
    --warmup 15 ^
    --epochs 10 ^
    --fine-tune 20 ^
    --use-aptos2019 --use-idrid --use-messidor --use-aptos2019-test-pl1 ^
    -f 3 --seed 100 ^
    -v --criterion-ord cauchy -c models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold3_heuristic_sinoussi.pth