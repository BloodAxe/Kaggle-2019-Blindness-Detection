import torch
import torch.nn.functional as F


def test_average_softmax_probs():
    logits = torch.tensor([[1, 6, 4, -3],
                           [2, 5, 6, 0]]).float()
    probs = logits.softmax(dim=1)
    print('Probs', probs)
    print('Mean probs', torch.mean(probs, dim=0))
    log_probs = F.log_softmax(logits, dim=1)
    print('Mean log', log_probs.mean(dim=0).exp())
