REM FineTune



python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 0 -t pretrained/806_813/seresnext50d_gwap/eloquent_yonath/fold0/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold0_eloquent_yonath.pth

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 1 -t pretrained/806_813/seresnext50d_gwap/eloquent_yonath/fold1/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold1_eloquent_yonath.pth

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 2 -t pretrained/806_813/seresnext50d_gwap/eloquent_yonath/fold2/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold2_eloquent_yonath.pth

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 3 -t pretrained/806_813/seresnext50d_gwap/eloquent_yonath/fold3/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold3_eloquent_yonath.pth


REM FineTune

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 0 -t pretrained/Aug14_11_10/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold0_zen_golick/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold0_zen_golick.pth

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 1 -t pretrained/Aug14_11_10/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold1_zen_golick/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold1_zen_golick.pth

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 2 -t pretrained/Aug14_11_10/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold2_zen_golick/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold2_zen_golick.pth

python train_ord.py -m seresnext50d_gwap -a medium -w 4 -b 64 -e 0 -ft 25 -o AdamW -wd 1e-4 -s multistep -lr 3e-4^
    --use-idrid --use-messidor --use-aptos2019 -v^
    --criterion-cls focal_kappa 0.5 --criterion-reg mse 0.5 -d 0.5^
    -f 3 -t pretrained/Aug14_11_10/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold3_zen_golick/checkpoints/seresnext50d_gwap_512_medium_aptos2019_messidor_idrid_fold3_zen_golick.pth
