import torch
import numpy as np
#
# checkpoints = [
#     'runs/classification/cls_resnet18/fold_0/Jul08_14_51_ce/checkpoints/fold0_best.pth',
#     'runs/classification/cls_resnet18/fold_1/Jul08_16_13_ce/checkpoints/fold1_best.pth',
#     'runs/classification/cls_resnet18/fold_2/Jul09_00_19_ce/checkpoints/fold2_best.pth',
#     'runs/classification/cls_resnet18/fold_3/Jul09_01_29_ce/checkpoints/fold3_best.pth'
# ]

# checkpoints = [
#     'runs/regression/reg_resnet18/fold_3/Jul09_18_20_clipped_mse/checkpoints/fold3_best.pth',
#     'runs/regression/reg_resnet18/fold_2/Jul09_16_32_clipped_mse/checkpoints/fold2_best.pth',
#     'runs/regression/reg_resnet18/fold_1/Jul09_14_57_clipped_mse/checkpoints/fold1_best.pth',
#     'runs/regression/reg_resnet18/fold_0/Jul09_13_44_clipped_mse/checkpoints/fold0_best.pth'
# ]

checkpoints = [
    'runs/regression/reg_resnext50/fold_0/Jul13_22_06_wing_loss/checkpoints/reg_resnext50_fold0.pth',
    'runs/regression/reg_resnext50/fold_1/Jul14_11_44_wing_loss/checkpoints/reg_resnext50_fold1.pth',
]
cv = []

for checkpoint in checkpoints:
    checkpoint = torch.load(checkpoint)
    cv.append(checkpoint['valid_metrics']['kappa_score'])

print(np.mean(cv))
print(np.std(cv))
