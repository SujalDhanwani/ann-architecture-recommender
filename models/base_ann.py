import torch
import torch.nn as nn


class BaseANN(nn.Module):
    """
    Fully configurable ANN model.

    Parameters:
    - input_size
    - output_size
    - hidden_layers (list)
    - activation (str)
    - dropout (float)
    - init_type (str)
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=[64, 32],
        activation="relu",
        dropout=0.0,
        init_type="xavier"
    ):
        super(BaseANN, self).__init__()

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        self.init_type = init_type

        # ---------------- Activation Mapping ----------------
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        # ---------------- Hidden Layers ----------------
        self.layers = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # ---------------- Output Layer ----------------
        self.output_layer = nn.Linear(prev_size, output_size)

        # ---------------- Dropout ----------------
        self.dropout = nn.Dropout(dropout)

        # ---------------- Initialization ----------------
        self.apply(self._init_weights)

    # -----------------------------------------------------
    # Weight Initialization
    # -----------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            if self.init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)

            elif self.init_type == "he":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

            else:
                raise ValueError("Unsupported initialization type")

            nn.init.zeros_(m.bias)

    # -----------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

            if self.dropout_rate > 0:
                x = self.dropout(x)

        x = self.output_layer(x)

        return x
