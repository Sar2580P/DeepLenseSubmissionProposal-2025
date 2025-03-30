import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class ResidualBlock(nn.Module):
    """Residual block with channel attention"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out

class PixelShuffleUpscale(nn.Module):
    """Upscaling module using pixel shuffle"""
    def __init__(self, in_channels, scale_factor):
        super(PixelShuffleUpscale, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class SuperResolutionDecoder(nn.Module):
    """
    Modern Super Resolution Decoder Architecture
    
    Args:
        in_channels (int): Number of input channels
        super_res_H (int): Output height after super resolution
        super_res_W (int): Output width after super resolution
        in_H (int): Input height
        in_W (int): Input width
        hidden_channels (int, optional): Number of hidden channels. Defaults to 128.
        residual_blocks (int, optional): Number of residual blocks. Defaults to 16.
        final_channels (int, optional): Number of output channels. Defaults to same as input.
    """
    def __init__(self, in_channels, super_res_H, super_res_W, in_H, in_W, 
                 hidden_channels=128, residual_blocks=16, final_channels=None):
        super(SuperResolutionDecoder, self).__init__()
        
        if final_channels is None:
            final_channels = in_channels
            
        # Calculate upscale factor
        h_scale_factor = super_res_H / in_H
        w_scale_factor = super_res_W / in_W
        
        assert h_scale_factor == w_scale_factor, "Height and width scale factors must be equal"
        assert h_scale_factor == int(h_scale_factor), "Scale factor must be an integer"
        scale_factor = int(h_scale_factor)
        
        # Calculate number of upscale steps (n where 2^n = scale_factor)
        n_upscales = 0
        temp = scale_factor
        while temp > 1:
            if temp % 2 != 0:
                raise ValueError("Scale factor must be a power of 2")
            temp //= 2
            n_upscales += 1
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(residual_blocks)
        ])
        
        # Global residual connection conv
        self.global_res_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Progressive upsampling layers (each doubles the resolution)
        self.upsampling_layers = nn.ModuleList()
        
        current_channels = hidden_channels
        for _ in range(n_upscales):
            self.upsampling_layers.append(PixelShuffleUpscale(current_channels, scale_factor=2))
        
        # Final output layer
        self.final = nn.Conv2d(hidden_channels, final_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial feature extraction
        initial_features = self.initial(x)
        
        # Residual blocks
        res = initial_features
        for block in self.residual_blocks:
            res = block(res)
        
        # Global residual connection
        res = self.global_res_conv(res)
        res += initial_features
        
        # Progressive upsampling
        for upsampler in self.upsampling_layers:
            res = upsampler(res)
        
        # Final output
        out = self.final(res)
        
        return out


if __name__ == "__main__":    # Parameters
    batch_size = 4
    in_channels = 128
    in_H, in_W = 19, 19
    final_channels=1
    
    # The super-resolution dimensions (2^n times the input)
    super_res_H, super_res_W = 152, 152 # 4x upscaling (2^2)
    
    # Create a sample input tensor
    x = torch.randn(batch_size, in_channels, in_H, in_W)
    
    # Initialize the decoder
    decoder = SuperResolutionDecoder(
        in_channels=in_channels,
        super_res_H=super_res_H,
        super_res_W=super_res_W,
        in_H=in_H,
        in_W=in_W,
        hidden_channels=128,
        residual_blocks=10, 
        final_channels=final_channels
    )
    
    # Forward pass
    output = decoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output shape: (4, 64, 128, 128)
    assert output.shape == (batch_size, final_channels, super_res_H, super_res_W)
    
    from utils.utils import plot_model
    model = decoder
    input_size = (batch_size, in_channels, in_H, in_W)
    
    plot_model(input_size, model, "SuperResDecoder", depth=2)