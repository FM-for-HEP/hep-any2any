import torch
import numpy as np
import awkward as ak

class VarTransformation:
    '''
        trans: tranforming the quantities
            eg. x -> log(x), pow(e,m) etc
        scale: scaling the quantities
            eg. x -> (x - mean(x)) / std(x)
        forward: trans + scale
    '''

    def __init__(self, config):
        self.config = config
        self.scale_mode = config['scale_mode']
        self.transformation = config['transformation']
        self.eps = 1e-6



    def trans(self, x):
        if self.transformation == None:
            return x
        if self.transformation == 'pow(x,m)':
            return x ** self.config['m']
        if self.transformation == 'inv_sigmoid':
            if isinstance(x, torch.Tensor):                
                x = torch.clamp(x, 0, 1)
                x = (1 - 2*self.eps) * x + self.eps # [0,1] -> [eps, 1-eps]
                x = torch.log(x / (1 - x))
            elif isinstance(x, np.ndarray):
                x = np.clip(x, 0, 1)
                x = (1 - 2*self.eps) * x + self.eps # [0,1] -> [eps, 1-eps]
                x = np.log(x / (1 - x))
            elif isinstance(x, ak.Array):
                x_flattened = ak.flatten(x)
                x_flattened = np.clip(x_flattened, 0, 1)
                x_flattened = (1 - 2*self.eps) * x_flattened + self.eps # [0,1] -> [eps, 1-eps]
                x_flattened = np.log(x_flattened / (1 - x_flattened))
                x = ak.unflatten(x_flattened, ak.num(x))
            else:
                raise ValueError(f'Unsupported type {type(x)} for transformation {self.transformation}')
            return x


    def inv_trans(self, x):
        if self.transformation == None:
            return x
        if self.transformation == 'pow(x,m)':
            return x ** (1 / self.config['m'])
        if self.transformation == 'inv_sigmoid':
            if isinstance(x, torch.Tensor):
                x = 1 / (1 + torch.exp(-x)) # sigmoid
                x = (x - self.eps) / (1 - 2*self.eps)
            else:
                x = 1 / (1 + np.exp(-x)) # sigmoid
                x = (x - self.eps) / (1 - 2*self.eps)
            return x



    def scale(self, x):
        if self.scale_mode == None:
            return x
        
        elif self.scale_mode == 'min_max':
            targmin, targmax = self.config['range']
            assert targmin < targmax, f'Invalid range {self.config["range"]}'
            return (x - self.config['min']) / (self.config['max'] - self.config['min']) \
                * (targmax - targmin) + targmin            
            
        elif self.scale_mode == 'standard':
            return (x - self.config['mean']) / self.config['std']
    
    def inv_scale(self, x):
        if self.scale_mode == None:
            return x
        
        elif self.scale_mode == 'min_max':
            targmin, targmax = self.config['range']
            assert targmin < targmax, f'Invalid range {self.config["range"]}'
            return (x - targmin) / (targmax - targmin) * \
                (self.config['max'] - self.config['min']) + self.config['min']
            
        elif self.scale_mode == 'standard':
            return x * self.config['std'] + self.config['mean']



    def forward(self, x):
        x = self.trans(x)
        x = self.scale(x)
        return x
    


    def inverse(self, x):
        x = self.inv_scale(x)
        x = self.inv_trans(x)
        return x

