from diffusion.architecture.model import CustomGaussianDiffusion
from diffusion.architecture.modules import Unet
from utils.utils import read_yaml

config = read_yaml('diffusion/config.yaml')
tr_config = config['train_config']
data_config = config['data_config']

unet = Unet(**config['UNet_params'])
if tr_config['model_name'].lower()=='Vanilla_Gaussian_Diffusion'.lower():
    model = CustomGaussianDiffusion(unet, **config['GaussianDiffusion_params'])
else:
    raise ValueError(f"Unknown model name: {tr_config['model_name']}")

# input_ = torch.randn(1, 1, 152, 152)

# output = model(input_)
# print(output.shape)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

# Example usage
print(f"Total Trainable Parameters: {count_parameters(model)} million")

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Includes batch norm, etc.
    total_size = (param_size + buffer_size) / (1024**2)  # Convert bytes to MB
    return total_size

# Example usage
print(f"Model Size: {get_model_size(model):.2f} MB")

