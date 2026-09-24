import torch
import torch.nn as nn



class PosTokenizer(nn.Module):
    def __init__(self): # , device):
        '''
        Args:
            config: config for the pos tokenizer
        '''
        super().__init__()
        self.eta_bins = 1024
        self.phi_bins = 1024
        self.n_quantizer = 3
        
        # Define grid ranges
        self.eta_min, self.eta_max = -3.0, 3.0
        self.phi_min, self.phi_max = -1.0, 1.0  # cos/sin range
        
        # Pre-compute bin edges
        self.register_buffer('eta_edges', torch.linspace(self.eta_min, self.eta_max, self.eta_bins + 1))
        self.register_buffer('cosphi_edges', torch.linspace(self.phi_min, self.phi_max, self.phi_bins + 1))
        self.register_buffer('sinphi_edges', torch.linspace(self.phi_min, self.phi_max, self.phi_bins + 1))

        # Pre-compute bin widths for efficiency
        self.eta_width = (self.eta_max - self.eta_min) / self.eta_bins
        self.cosphi_width = (self.phi_max - self.phi_min) / self.phi_bins
        self.sinphi_width = (self.phi_max - self.phi_min) / self.phi_bins

    def forward(self, eta_cosphi_sinphi):
        """
        Fast vectorized index computation
        Args:
            eta_cosphi_sinphi: torch tensors of shape (..., 3)
        Returns:
            indices: torch tensor of shape (..., 3) with [eta_idx, cosphi_idx, sinphi_idx]
        """
        # Clamp values to valid ranges
        eta = torch.clamp(eta_cosphi_sinphi[..., 0], self.eta_min, self.eta_max - 1e-6)
        cosphi = torch.clamp(eta_cosphi_sinphi[..., 1], self.phi_min, self.phi_max - 1e-6)
        sinphi = torch.clamp(eta_cosphi_sinphi[..., 2], self.phi_min, self.phi_max - 1e-6)

        # Compute indices using vectorized operations
        eta_idx = ((eta - self.eta_min) / self.eta_width).long()
        cosphi_idx = ((cosphi - self.phi_min) / self.cosphi_width).long()
        sinphi_idx = ((sinphi - self.phi_min) / self.sinphi_width).long()
        
        # Stack indices
        indices = torch.stack([eta_idx, cosphi_idx, sinphi_idx], dim=-1)
        return indices
    
    def decode(self, indices):
        """
        Decode grid indices back to center coordinates
        Args:
            indices: torch tensor of shape (..., 3) with [eta_idx, cosphi_idx, sinphi_idx]
        Returns:
            eta, cosphi, sinphi: torch tensors of shape (..., )
        """
        eta_idx, cosphi_idx, sinphi_idx = indices[..., 0], indices[..., 1], indices[..., 2]
        
        eta = self.eta_min + (eta_idx.float() + 0.5) * self.eta_width
        cosphi = self.phi_min + (cosphi_idx.float() + 0.5) * self.cosphi_width
        sinphi = self.phi_min + (sinphi_idx.float() + 0.5) * self.sinphi_width
        
        eta_cosphi_sinphi = torch.stack([eta, cosphi, sinphi], dim=-1)
        return eta_cosphi_sinphi