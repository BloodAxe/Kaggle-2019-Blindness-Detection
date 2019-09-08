REM Pretrain model to start with something more or less accurate

python train_classifier_baseline.py -m resnet18_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v --criterion-cls hybrid_kappa

python train_classifier_baseline.py -m resnet34_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v --criterion-cls hybrid_kappa

python train_classifier_baseline.py -m resnet50_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v --criterion-cls hybrid_kappa

python train_classifier_baseline.py -m resnet152_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 6 -o AdamW -lr 1e-4 -wd 1e-4 -v --criterion-cls hybrid_kappa

python train_classifier_baseline.py -m seresnext50_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v --criterion-cls hybrid_kappa

python train_classifier_baseline.py -m seresnext101_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 6 -o AdamW -lr 1e-4 -wd 1e-4 -v --criterion-cls hybrid_kappa


REM Regression

REM Pretrain model to start with something more or less accurate

python train_regression_baseline.py -m resnet18_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v

python train_regression_baseline.py -m resnet34_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v

python train_regression_baseline.py -m resnet50_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v

python train_regression_baseline.py -m resnet152_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 6 -o AdamW -lr 1e-4 -wd 1e-4 -v

python train_regression_baseline.py -m seresnext50_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 8 -o AdamW -lr 1e-4 -wd 1e-4 -v

python train_regression_baseline.py -m seresnext101_gap^
    -f 0 --use-aptos2019 --use-idrid --use-messidor -e 50^
    -a light -d 0.5 -b 6 -o AdamW -lr 1e-4 -wd 1e-4 -v