from torch import nn



class StandardMLP(nn.Module):
    def __init__(self, dim_in, dim_out, widths):
        super(StandardMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.widths = widths
        self.linear_in = nn.Linear(self.dim_in, self.widths[0])
        self.linear_out = nn.Linear(self.widths[-1], self.dim_out)
        self.layers = []
        self.layer_norms = []
        for i in range(len(self.widths) - 1):
            self.layers.append(nn.Linear(self.widths[i], self.widths[i + 1]))
            self.layer_norms.append(nn.LayerNorm(widths[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.layernorms = nn.ModuleList(self.layer_norms)

    def forward(self, x):
        z = self.linear_in(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            z = norm(z)
            z = layer(z)
        out = self.linear_out(z)
        return out




class BottleneckBlock(nn.Module):
    def __init__(self, thin, wide):
        super(BottleneckBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(thin, wide), 
            nn.GELU(), 
            nn.Linear(wide, thin)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class BottleneckMLP(nn.Module):
    def __init__(self, dim_in, dim_out, block_dims):
        super(BottleneckMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.block_dims = block_dims

        self.linear_in = nn.Linear(self.dim_in, self.block_dims[0][1])
        self.linear_out = nn.Linear(self.block_dims[-1][1], self.dim_out)
        blocks = []
        layernorms = []

        for block_dim in self.block_dims:
            wide, thin = block_dim
            blocks.append(BottleneckBlock(thin=thin, wide=wide))
            layernorms.append(nn.LayerNorm(thin))

        self.blocks = nn.ModuleList(blocks)
        self.layernorms = nn.ModuleList(layernorms)

    def forward(self, x):
        x = self.linear_in(x)

        for block, norm in zip(self.blocks, self.layernorms):
            x = x + block(norm(x))

        out = self.linear_out(x)
        return out




def build_mlp(is_bottleneck, input_size=None, output_size= None,
        num_blocks=3, thin=64, expansion_factor=4, **kwargs) -> nn.Module:
    """Constructs an MLP model
    "Scaling MLPs: A Tale of Inductive Bias" (https://arxiv.org/abs/2306.13575).
    
    Args:
        is_bottleneck: 
        input_size: Input dimensionality. If None, defaults to MLP dimension.
        output_size: Output dimensionality. If None, defaults to MLP dimension.

    Returns:
        MLP model.
    """
    dim_in = input_size or thin
    dim_out = output_size or thin

    if is_bottleneck:
        blocks = [[expansion_factor * thin, thin] for _ in range(num_blocks)]
        return BottleneckMLP(
            dim_in=dim_in,
            dim_out=dim_out,
            block_dims=blocks,
        )

    else:
        blocks = [thin for _ in range(num_blocks)]
        return StandardMLP(
            dim_in=dim_in,
            dim_out=dim_out,
            widths=blocks,
        )