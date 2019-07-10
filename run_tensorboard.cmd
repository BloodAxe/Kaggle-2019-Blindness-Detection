@call c:\Anaconda3\Scripts\activate.bat tb
set CUDA_VISIBLE_DEVICES=
tensorboard --logdir runs/regression/reg_stn_resnet18 --host 0.0.0.0 --port 5555