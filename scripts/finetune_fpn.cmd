python train_ord.py -m seresnext50_fpn -a medium --size 512 -b 8 -v -o RAdam -wd 1e-3 -d 0.5 -lr 1e-4 -w 8^
    --use-aptos2019 --use-idrid --use-messidor -f 0 -f 1 -f 2 -f 3 --criterion-ord huber -t pretrain/seresnext50d_gap_pretrain.pth --warmup 10 -e 75 -ft 15
