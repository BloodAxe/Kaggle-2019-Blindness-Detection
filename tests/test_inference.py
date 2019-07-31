import cv2
import torch
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

from retinopathy.augmentations import get_test_transform, AddMildDR
from retinopathy.dataset import get_class_names
from retinopathy.factory import get_model


def test_inference():
    model_checkpoint = '../runs/reg/reg_resnet34_rms/Jul30_22_13/reg_resnet34_rms_512_hard_wing_loss_aptos2019_messidor_idridfold0_silly_lichterman/checkpoints/best.pth'
    checkpoint = torch.load(model_checkpoint)
    model_name = checkpoint['checkpoint_data']['cmd_args']['model']

    num_classes = len(get_class_names())
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    for image_fname in [
        # '4_left.png',
        # '35_left.png',
        '44_right.png',
        '68_right.png',
        # '92_left.png'
    ]:
        transform = get_test_transform(image_size=(512,512), crop_black=True)

        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_transformed = transform(image=image)['image']
        image_transformed = tensor_from_rgb_image(image_transformed).unsqueeze(0)

        with torch.no_grad():
            model = model.eval().cuda()
            predictions = model(image_transformed.cuda())
            print(predictions['logits'].softmax(dim=1))
            print(predictions['regression'])

        add_mild_dr = AddMildDR(p=1)
        data = add_mild_dr(image=image, diagnosis=0)
        image_transformed = transform(image=data['image'])['image']
        image_transformed = tensor_from_rgb_image(image_transformed).unsqueeze(0)

        with torch.no_grad():
            model = model.eval().cuda()
            predictions = model(image_transformed.cuda())
            print(predictions['logits'].softmax(dim=1))
            print(predictions['regression'])
