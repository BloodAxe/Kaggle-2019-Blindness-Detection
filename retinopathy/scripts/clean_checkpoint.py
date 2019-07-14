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


clean_checkpoint('runs/regression/reg_resnext50/fold_0/Jul13_22_06_wing_loss/checkpoints/reg_resnext50_fold0.pth')
clean_checkpoint('runs/regression/reg_resnext50/fold_1/Jul14_11_44_wing_loss/checkpoints/reg_resnext50_fold1.pth')
