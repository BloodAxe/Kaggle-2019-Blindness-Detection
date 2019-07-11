from retinopathy.lib.factory import get_model
from catalyst.contrib import registry


@registry.Model
def reg_resnet18(**kwargs):
    return get_model('reg_resnet18', num_classes=1, **kwargs)
