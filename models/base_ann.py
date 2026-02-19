import torch
import torch.nn as nn


class BaseANN(nn.Module):
    """
    Fully configurable Artificial Neural Network.

    Parameters:
    - input_size (int)
    - output_size (int)
    - hidden_layers (list of int)
    - activation (str)
    - dropout (float)
    - init_type (str)
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=None,
        activation="relu",
        dropout=0.0,
        init_type="xavier"
    ):
        super(BaseANN, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        self.init_type = init_type
        self.activation_name = activation.lower()

        # ---------------- Activation Mapping ----------------
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }

        if self.activation_name not in activation_map:
            raise ValueError("Unsupported activation function")

        self.activation = activation_map[self.activation_name]

        # ---------------- Build Layers ----------------
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.hidden_block = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

        # Initialize weights
        self.apply(self._init_weights)

    # -----------------------------------------------------
    # Weight Initialization
    # -----------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            if self.init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)

            elif self.init_type == "he":
                # Use correct nonlinearity for He init
                nonlinearity = "relu" if self.activation_name == "relu" else "linear"
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

            else:
                raise ValueError("Unsupported initialization type")

            nn.init.zeros_(m.bias)

    # -----------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------
    def forward(self, x):
        x = self.hidden_block(x)
        x = self.output_layer(x)
        return x
