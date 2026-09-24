import torch.nn as nn

class Upsampler(nn.Module):
    def __init__(self, in_channels, n_upsample):
        """
        2x Upsampling n_upsample times.
        Args:
            in_channels (int): Channels in the input image grid.
        """
        super().__init__()
        
        # --- Input is (bs, in_channels, 8, 8) ---
        self.upsampler = []
        for i in range(n_upsample):
            in_ch = in_channels // 2 if i > 0 else in_channels
            out_ch = in_channels // 2
            self.upsampler += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=out_ch),
                nn.LeakyReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=out_ch),
                nn.LeakyReLU()
            ]

        self.upsampler = nn.Sequential(*self.upsampler)

    def forward(self, x):
        return self.upsampler(x)
