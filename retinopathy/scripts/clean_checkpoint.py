import torch


def clean_checkpoint(fname):
    checkpoint = torch.load(fname)

    keys = ['criterion_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'oof_predictions'
            ]
    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, fname)


clean_checkpoint('runs/classification/cls_resnext50/fold_0/Jul14_05_32_ce_fp16/checkpoints/cls_resnext50_fold0.pth')
clean_checkpoint('runs/classification/cls_resnext50/fold_1/Jul14_08_21_ce_fp16/checkpoints/cls_resnext50_fold1.pth')
clean_checkpoint('runs/classification/cls_resnext50/fold_2/Jul14_11_11_ce_fp16/checkpoints/cls_resnext50_fold2.pth')
clean_checkpoint('runs/classification/cls_resnext50/fold_3/Jul14_14_02_ce_fp16/checkpoints/cls_resnext50_fold3.pth')
