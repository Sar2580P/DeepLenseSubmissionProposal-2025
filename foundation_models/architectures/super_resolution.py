import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from foundation_models.architectures.super_res_decoder_modules import SuperResolutionDecoder

class SuperResolutionAE(nn.Module):
    def __init__(
        self,
        encoder,
        patch_height=16, 
        patch_width=16, 
        high_res_height=128, 
        high_res_width=128,
        low_res_height = 64, 
        low_res_width =64 ,
        out_channels=1,
        shift=0.0 ,
        scale=1.0
        ):
        super().__init__()
        # Save the encoder (a Vision Transformer to be trained)
        self.encoder = encoder
        # Extract the number of patches and the encoder's dimensionality from the positional embeddings
        encoder_dim = encoder.pos_embedding.shape[-1]

        # Separate the patch embedding layers from the encoder
        # The first layer converts the image into patches
        self.to_patch = encoder.to_patch_embedding[0]
        # The remaining layers embed the patches
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        
        self.reshape_patches_to_img = Rearrange(
            'b (h w) c -> b c h w',
            h=low_res_height// patch_height,
            w=low_res_width // patch_width,
        )
        
        self.super_res_decoder = SuperResolutionDecoder(
                                    in_channels=encoder_dim,
                                    super_res_H=high_res_height,
                                    super_res_W=high_res_width,
                                    in_H=low_res_height// patch_height,
                                    in_W=low_res_width // patch_width,
                                    hidden_channels=128,
                                    residual_blocks=10, 
                                    final_channels=out_channels
                                )
        
        self.shift = shift
        self.scale = scale

    def compute_loss(self, pred_super_res, high_res_img):
        # Compute the reconstruction loss (mean squared error)
        weight_mask = (high_res_img + self.shift) * self.scale
        weighted_recon_loss = F.mse_loss(pred_super_res, high_res_img, reduction="mean", weight=weight_mask)
        return weighted_recon_loss
    

    def forward(self, img):
        device = img.device

        # Convert the input image into patches
        patches = self.to_patch(img)  # Shape: (batch_size, num_patches, patch_size)
        _, num_patches, *_ = patches.shape

        # Embed the patches using the encoder's patch embedding layers
        tokens = self.patch_to_emb(patches)  # Shape: (batch_size, num_patches, encoder_dim)
        # Add positional embeddings to the tokens
        if self.encoder.pool == "cls":
            # If using CLS token, skip the first positional embedding
            tokens += self.encoder.pos_embedding[:, 1 : num_patches + 1]
        elif self.encoder.pool == "mean":
            # If using mean pooling, use all positional embeddings
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)
            
        # Encode the unmasked tokens using the encoder's transformer
        encoded_tokens = self.encoder.transformer(tokens)
        
        # convert patched to image
        encoder_features = self.reshape_patches_to_img(encoded_tokens)
        
        pred_super_res = self.super_res_decoder(encoder_features)
        return  encoder_features[:, 0, :, :] , pred_super_res
    


if __name__=="__main__":
   from utils.utils import plot_model, read_yaml
   from foundation_models.architectures.vit import ViT
   from foundation_models.architectures.super_resolution import SuperResolutionAE
   config = read_yaml('foundation_models/configs/super_res_config.yaml')
   encoder = ViT(**config['ViT_params'])
   model = SuperResolutionAE(encoder=encoder, **config["SuperRes_params"])
   input_size = (8, 1, 76, 76)
    
   plot_model(input_size, model, "SuperResolutionAE", depth=2)

