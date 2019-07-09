import torch
import numpy as np

checkpoints = [
    'runs/classification/cls_resnet18/fold_0/Jul08_14_51_ce/checkpoints/fold0_best.pth',
    'runs/classification/cls_resnet18/fold_1/Jul08_16_13_ce/checkpoints/fold1_best.pth',
    'runs/classification/cls_resnet18/fold_2/Jul09_00_19_ce/checkpoints/fold2_best.pth',
    'runs/classification/cls_resnet18/fold_3/Jul09_01_29_ce/checkpoints/fold3_best.pth'
]

cv = []

for checkpoint in checkpoints:
    checkpoint = torch.load(checkpoint)
    cv.append(checkpoint['valid_metrics']['kappa_score'])

print(np.mean(cv))
print(np.std(cv))
