import cv2
import pytest
import torch
from pytorch_toolbelt.utils.torch_utils import count_parameters, tensor_from_rgb_image

from retinopathy.augmentations import get_test_transform
from retinopathy.factory import get_model
from retinopathy.models.heads.rnn import LSTMBottleneck


@pytest.mark.parametrize('model_name',
                         [
                             # 'resnet18_fpn',
                             # 'resnet18_gap',
                             # 'resnet18_max',
                             # 'resnet18_gwap',
                             'pnasnet5_gapv2'
                         ])
@torch.no_grad()
def test_cls_models(model_name):
    model = get_model(model_name=model_name, num_classes=4).eval()
    print(model_name, count_parameters(model))
    return
    x = torch.rand((1, 3, 224, 224))
    output = model(x)
    assert output['logits'].size(1) == 4
    assert output['features'].size(1) == model.features_size

    print(model_name, count_parameters(model))


def test_lstm_bottleneck():
    net = LSTMBottleneck(512, 64)
    x = torch.rand((4, 512, 32, 32))
    y = net(x)
    assert y.size(1) == 64
