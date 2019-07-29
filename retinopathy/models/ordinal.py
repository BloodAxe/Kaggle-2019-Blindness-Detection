import torch
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn


class LogisticCumulativeLink(nn.Module):
    """
    Converts a single number to the proportional odds of belonging to a class.
    Parameters
    ----------
    num_classes : int
        Number of ordered classes to partition the odds into.
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, num_classes: int,
                 init_cutpoints: str = 'ordered') -> None:
        assert num_classes > 2, (
            'Only use this model if you have 3 or more classes'
        )
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == 'ordered':
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == 'random':
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
            raise ValueError(f'{init_cutpoints} is not a valid init_cutpoints '
                             f'type')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Equation (11) from
        "On the consistency of ordinal regression methods", Pedregosa et. al.
        """
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
            sigmoids[:, [0]],
            link_mat,
            (1 - sigmoids[:, [-1]])
        ),
            dim=1
        )
        return link_mat


class OrdinalEncoderHeadModel(nn.Module):
    def __init__(self, encoder: EncoderModule, head, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.link = LogisticCumulativeLink(num_classes,
                                           init_cutpoints='ordered')
    @property
    def features_size(self):
        return self.head.features_size

    def forward(self, input):
        feature_maps = self.encoder(input)
        features, logits = self.head(feature_maps)
        logits = self.link(logits)
        return {'features': features, 'logits': logits}
