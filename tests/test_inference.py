import cv2
import torch
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
import pandas as pd

from retinopathy.augmentations import get_test_transform, AddMicroaneurisms
from retinopathy.dataset import get_class_names
from retinopathy.factory import get_model
from retinopathy.inference import run_model_inference
from retinopathy.models.common import regression_to_class


def test_inference():
    model_checkpoint = '../pretrained/seresnext50_gap_512_medium_aptos2019_idrid_fold0_hopeful_easley.pth'
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
        transform = get_test_transform(image_size=(512, 512), crop_black=True)

        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_transformed = transform(image=image)['image']
        image_transformed = tensor_from_rgb_image(image_transformed).unsqueeze(0)

        with torch.no_grad():
            model = model.eval().cuda()
            predictions = model(image_transformed.cuda())
            print(predictions['logits'].softmax(dim=1))
            print(predictions['regression'])

        add_mild_dr = AddMicroaneurisms(p=1)
        data = add_mild_dr(image=image, diagnosis=0)
        image_transformed = transform(image=data['image'])['image']
        image_transformed = tensor_from_rgb_image(image_transformed).unsqueeze(0)

        with torch.no_grad():
            model = model.eval().cuda()
            predictions = model(image_transformed.cuda())
            print(predictions['logits'].softmax(dim=1))
            print(predictions['regression'])




def test_inference_pd():
    test_csv = pd.read_csv('data/aptos-2019/test.csv')

    checkpoints = [
        'runs/Aug09_16_47/seresnext50_gap/512/medium/hopeful_easley/fold0/checkpoints/seresnext50_gap_512_medium_aptos2019_idrid_fold0_hopeful_easley.pth',
    ]

    for checkpoint in checkpoints:
        predictions = run_model_inference(model_checkpoint=fs.auto_file(checkpoint),
                                          test_csv=test_csv,
                                          images_dir='test_images_768',
                                          data_dir='data/aptos-2019',
                                          apply_softmax=True,
                                          need_features=False,
                                          batch_size=8)

        predictions['diagnosis'] = predictions['diagnosis'].apply(regression_to_class).apply(int)
        submission = predictions[['id_code','diagnosis']]
        print(submission)