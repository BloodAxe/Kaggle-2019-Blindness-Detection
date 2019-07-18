import cv2
import pytest
import torch
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import count_parameters, tensor_from_rgb_image

from retinopathy.lib.augmentations import get_test_aug
from retinopathy.lib.factory import get_model


@pytest.mark.parametrize('model_name',
                         [
                             'cls_resnet18_gap',
                             'cls_resnet18_gmp',
                             'cls_resnet18_gwap',
                             'cls_resnet18_ocp',
                             'cls_resnet18_rms',
                             'cls_resnet18_maxavg',
                         ])
@torch.no_grad()
def test_cls_models(model_name):
    model = get_model(model_name=model_name, num_classes=4).eval().cuda()
    x = torch.rand((1, 3, 512, 512)).cuda()
    output = model(x)
    assert output['logits'].size(1) == 4
    assert output['features'].size(1) == 512

    print(model_name, count_parameters(model))


@pytest.mark.parametrize('model_name',
                         [
                             'reg_resnet18_gap',
                             'reg_resnet18_gmp',
                             'reg_resnet18_gwap',
                             'reg_resnet18_ocp',
                             'reg_resnet18_rms',
                             'reg_resnet18_maxavg',
                         ])
@torch.no_grad()
def test_reg_models(model_name):
    for image_fname in [
        '4_left.png',
        '35_left.png',
        '44_right.png',
        '68_right.png',
        '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = get_test_aug(image_size=(512, 512), crop_black=True)

        input_image = transform(image=image)['image']
        model = get_model(model_name=model_name, num_classes=5).eval().cuda()
        x = tensor_from_rgb_image(input_image).unsqueeze(0).cuda()

        output = model(x)
        print(model_name, count_parameters(model))
        print(output['logits'])

        assert output['logits'].size(0) == 1
        assert output['features'].size(1) == 512



@pytest.mark.parametrize('model_name',
                         [
                             'ord_resnet18_gap',
                             'ord_resnet18_gmp',
                             'ord_resnet18_gwap',
                             'ord_resnet18_ocp',
                             'ord_resnet18_rms',
                             'ord_resnet18_maxavg',
                         ])
@torch.no_grad()
def test_ord_models(model_name):
    model = get_model(model_name=model_name, num_classes=4).eval().cuda()
    x = torch.rand((1, 3, 512, 512)).cuda()

    output = model(x)
    assert output['logits'].size(1) == 4
    assert output['features'].size(1) == 512

    print(model_name, count_parameters(model))
