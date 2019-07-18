import torch


def clean_checkpoint(src_fname, dst_fname):
    checkpoint = torch.load(src_fname)

    keys = [
        'criterion_state_dict',
        'optimizer_state_dict',
        'scheduler_state_dict',
    ]

    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, dst_fname)


checkpoints = [
    'runs/regression/reg_resnext101_multi/fold_3/Jul16_22_24_wing_loss_fp16_fast/checkpoints/reg_resnext101_multi_fold3.pth',
    'runs/regression/reg_resnext101_multi/fold_2/Jul16_21_02_wing_loss_fp16_fast/checkpoints/reg_resnext101_multi_fold2.pth',
    'runs/regression/reg_resnext101_multi/fold_1/Jul16_18_08_wing_loss_fp16_fast/checkpoints/reg_resnext101_multi_fold1.pth',
    'runs/regression/reg_resnext101_multi/fold_0/Jul16_15_57_wing_loss_fp16_fast/checkpoints/reg_resnext101_multi_fold0.pth'
]

for c in checkpoints:
    clean_checkpoint(c)
