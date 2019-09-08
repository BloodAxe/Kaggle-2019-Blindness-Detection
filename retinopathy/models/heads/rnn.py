from pytorch_toolbelt.modules.coord_conv import append_coords
from torch import nn


class LSTMBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_size, dropout=0.1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels + 3,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, input):
        input = append_coords(input, with_r=True)

        batch_size = input.size(0)
        in_channels = input.size(1)
        rows = input.size(2)
        cols = input.size(3)
        input = input.permute((0, 3, 1, 2)).reshape(batch_size, -1, in_channels)

        self.lstm.flatten_parameters()

        lstm_out, hidden = self.lstm(input)
        lstm_out = lstm_out.view(batch_size, rows * cols, 2, -1)  # (batch, seq_len, num_directions, hidden_size)

        lstm_left = lstm_out[:, :, 0, :]
        lstm_right = lstm_out[:, :, 1, :]
        lstm_out = lstm_left + lstm_right
        last_out = lstm_out[:, -1, :]  # Many to one
        return last_out


class RNNHead(nn.Module):
    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1] // 8
        self.rnn_pool = LSTMBottleneck(feature_maps[-1] + 3, self.features_size, dropout=dropout)

        self.logits = nn.Linear(self.features_size, num_classes)

        self.regression = nn.Sequential(
            nn.Linear(self.features_size, self.features_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.features_size, self.features_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.features_size, 1),
        )

        self.ordinal = nn.Sequential(
            nn.Linear(self.features_size, self.features_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.features_size, num_classes - 1),
            nn.Sigmoid())

    def forward(self, feature_maps):
        # Take last feature map
        features = feature_maps[-1]
        features = append_coords(features, with_r=True)
        features = self.rnn_pool(features)

        # Squeeze to num_classes
        logits = self.logits(features)
        regression = self.regression(features)
        ordinal = self.ordinal(features).sum(dim=1)

        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            'ordinal': ordinal
        }
