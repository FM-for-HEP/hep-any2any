from torch import nn

class Dense(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
        activation = "ReLU",
        final_activation = None,
        norm_layer = None,
        norm_final_layer = False,
        dropout = 0.0,
        context_size = 0,
    ):
        """A simple fully connected feed forward neural network.

        Parameters
        ----------
        input_size : int
            Input size
        output_size : int
            Output size
        hidden_layers : list
            Number of nodes per layer
        activation : str
            Activation function for hidden layers, by default "ReLU"
        final_activation : str, optional
            Activation function for the output layer, by default None
        norm_layer : str, optional
            Normalisation layer, by default None
        norm_final_layer : bool, optional
            Whether to use normalisation on the final layer, by default False
        dropout : float, optional
            Apply dropout with the supplied probability, by default 0.0
        context_size : int
            Must be 0; the key appears in the model configs of the released tokenisers
        """
        super().__init__()
        if context_size:
            raise ValueError("Dense does not take context inputs (context_size must be 0)")

        # Save the networks input and output sizes
        self.input_size = input_size
        self.output_size = output_size

        # build nodelist
        node_list = [input_size, *hidden_layers, output_size]

        # input and hidden layers
        layers = []

        num_layers = len(node_list) - 1
        for i in range(num_layers):
            is_final_layer = i == num_layers - 1

            # normalisation first
            if norm_layer and (norm_final_layer or not is_final_layer):
                layers.append(getattr(nn, norm_layer)(node_list[i], elementwise_affine=False))

            # then dropout
            if dropout and (norm_final_layer or not is_final_layer):
                layers.append(nn.Dropout(dropout))

            # linear projection
            layers.append(nn.Linear(node_list[i], node_list[i + 1]))

            # activation
            if not is_final_layer:
                layers.append(getattr(nn, activation)())

            # final layer: return logits by default, otherwise apply activation
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        # build the net
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
