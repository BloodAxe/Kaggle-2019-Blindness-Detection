import pytest
import torch
from pytorch_toolbelt.utils.torch_utils import count_parameters

from retinopathy.lib.factory import get_model


@pytest.mark.parametrize('model_name',
                         [
                             'cls_resnet18_gap',
                             'cls_resnet18_gmp',
                             'cls_resnet18_gwap',
                             'cls_resnet18_ocp',
                             'cls_resnet18_rms',
                             'cls_resnet18_maxavg',

                             'reg_resnet18_gap',
                             'reg_resnet18_gmp',
                             'reg_resnet18_gwap',
                             'reg_resnet18_ocp',
                             'reg_resnet18_rms',
                             'reg_resnet18_maxavg',
                         ])
@torch.no_grad()
def test_models(model_name):
    model = get_model(model_name=model_name, num_classes=4).eval().cuda()
    x = torch.rand((1, 3, 512, 512)).cuda()
    y = model(x)

    print(model_name, count_parameters(model))
