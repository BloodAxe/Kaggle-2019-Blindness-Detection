import torch
import torch.nn.functional as F
import pytest

def test_kl():
    logits = torch.tensor([
        [10, 1, 2, 3, 4]
    ]).float()

    target = logits.log_softmax(dim=1).exp()
    input = logits.log_softmax(dim=1)
    l = F.kl_div(input, target, reduction='batchmean')
    print(l)


    logits2 = torch.tensor([
        [1, 10, 2, 3, 4]
    ]).float()

    input2 = logits2.log_softmax(dim=1)
    l = F.kl_div(input2, target, reduction='batchmean')
    print(l)
