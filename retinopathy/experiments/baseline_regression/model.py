from catalyst.contrib import registry

from retinopathy.factory import get_model


@registry.Model
def reg_resnet50_rms(**kwargs):
    return get_model('reg_resnet50_rms', num_classes=5, **kwargs)
