
REM Pretrain model to start with something more or less accurate
python train_classifier_baseline.py -m seresnext50_max -a light -f 0 -b 60 --fp16 -o Adam -lr 1e-4\
  -e 20 -v --use-aptos2019 --use-idrid --use-messidor --warmup 20 --criterion ce -x cls_seresnext50_max_pretrain
sleep 5

