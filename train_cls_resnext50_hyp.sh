#!/usr/bin/env bash
python -m cls_resnext50_hyp -a hard -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam --balance -s multistep -lr 1e-4 -wd 1e-4 -e 100 -v
pause 5

python -m cls_resnext50_hyp -a hard -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam --balance -s multistep -lr 1e-4 -wd 1e-4 -e 100 -v --use-idrid
pause 5

python -m cls_resnext50_hyp -a hard -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam --balance -s multistep -lr 1e-4 -wd 1e-4 -e 100 -v --use-idrid --use-messidor
pause 5