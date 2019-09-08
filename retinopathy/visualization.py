import cv2
import numpy as np
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, to_numpy

from retinopathy.dataset import UNLABELED_CLASS
from retinopathy.models.regression import regression_to_class


def draw_classification_predictions(input: dict,
                                    output: dict,
                                    class_names,
                                    image_key='image',
                                    image_id_key='image_id',
                                    targets_key='targets',
                                    outputs_key='logits',
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)):
    images = []

    for image, target, image_id, logits in zip(input[image_key],
                                               input[targets_key],
                                               input[image_id_key],
                                               output[outputs_key]):
        image = rgb_image_from_tensor(image, mean, std)
        num_classes = logits.size(0)
        target = int(to_numpy(target).squeeze(0))

        if num_classes == 1:
            logits = int(to_numpy(logits).squeeze(0) > 0)
        else:
            logits = np.argmax(to_numpy(logits))

        overlay = image.copy()

        if target != UNLABELED_CLASS:
            target_name = class_names[target]
        else:
            target_name = 'Unlabeled'

        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))
        cv2.putText(overlay, target_name, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 250, 0))
        if target == logits:
            cv2.putText(overlay, class_names[logits], (10, 45), cv2.FONT_HERSHEY_PLAIN, 1, (0, 250, 0))
        else:
            cv2.putText(overlay, class_names[logits], (10, 45), cv2.FONT_HERSHEY_PLAIN, 1, (250, 0, 0))

        images.append(overlay)

    return images


def draw_regression_predictions(input: dict,
                                output: dict,
                                class_names,
                                image_key='image',
                                image_id_key='image_id',
                                targets_key='targets',
                                outputs_key='regression',
                                unsupervised_label=None,
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)):
    images = []

    for i, (image, target, image_id) in enumerate(zip(input[image_key],
                                                      input[targets_key],
                                                      input[image_id_key])):
        diagnosis = output[outputs_key][i]
        image = rgb_image_from_tensor(image, mean, std)
        target = int(to_numpy(target).squeeze(0))
        predicted_target = int(regression_to_class(diagnosis))
        overlay = image.copy()

        if 'stn' in output:
            stn = rgb_image_from_tensor(output['stn'][i], mean, std)
            overlay = np.hstack((overlay, stn))

        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))
        if target != unsupervised_label:
            cv2.putText(overlay, f'{class_names[target]} ({target})', (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 250, 0))
        else:
            cv2.putText(overlay, f'Unlabeled ({target})', (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 250, 0))

        cv2.putText(overlay, f'{class_names[predicted_target]} ({predicted_target}/{float(diagnosis)})', (10, 45),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 250, 250))

        images.append(overlay)

    return images
