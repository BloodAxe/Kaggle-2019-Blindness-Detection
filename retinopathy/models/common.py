import numpy as np
import torch
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)




def regression_to_class(value: torch.Tensor, min=0, max=4, rounding_coefficients=None):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if isinstance(value, (int, float)):
        value = torch.tensor(value)

    if rounding_coefficients is None:
        value = torch.round(value)
        value = torch.clamp(value, min, max)
    else:
        rounded = torch.zeros(len(value))
        rounded[value < rounding_coefficients[0]] = 0
        rounded[(value >= rounding_coefficients[0]) & (value < rounding_coefficients[1])] = 1
        rounded[(value >= rounding_coefficients[1]) & (value < rounding_coefficients[2])] = 2
        rounded[(value >= rounding_coefficients[2]) & (value < rounding_coefficients[3])] = 3
        rounded[value >= rounding_coefficients[3]] = 4
        value = rounded.long()
    return value.long()





class EncoderHeadModel(nn.Module):
    def __init__(self, encoder: EncoderModule, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def features_size(self):
        return self.head.features_size

    def forward(self, image):
        feature_maps = self.encoder(image)
        result = self.head(feature_maps)
        return result

