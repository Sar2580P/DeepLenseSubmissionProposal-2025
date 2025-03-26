import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from foundation_models.architectures.mae import MAE

class SuperResolutionAE(MAE):
    def __init__(
        self, 
        patch_height=16, 
        patch_width=16, 
        high_res_height=128, 
        high_res_width=128, 
        **kwargs):
        super().__init__(**kwargs)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.high_res_height = high_res_height
        self.high_res_width = high_res_width

        # Upsampling layers for super-resolution
        self.upsample = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=patch_height, p2=patch_width),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),

            # Adaptive Upsampling
            nn.Upsample(size=(high_res_height // 2, high_res_width // 2), mode="bilinear", align_corners=False),
            nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),

            nn.Upsample(size=(high_res_height, high_res_width), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, img, high_res_img):
        device = img.device

        # Convert the input image into patches
        patches = self.to_patch(img)  # Shape: (batch_size, num_patches, patch_size)
        batch_size, num_patches, *_ = patches.shape

        # Embed the patches using the encoder's patch embedding layers
        tokens = self.patch_to_emb(patches)  # Shape: (batch_size, num_patches, encoder_dim)
        # Add positional embeddings to the tokens
        if self.encoder.pool == "cls":
            # If using CLS token, skip the first positional embedding
            tokens += self.encoder.pos_embedding[:, 1 : num_patches + 1]
        elif self.encoder.pool == "mean":
            # If using mean pooling, use all positional embeddings
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)
            
        masked_indices , unmasked_indices, num_masked = self.perform_masking(num_patches , device , batch_size)
        # Select the tokens corresponding to unmasked patches
        batch_range = torch.arange(batch_size, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # Select the original patches that are masked (for reconstruction loss)
        masked_patches = patches[batch_range, masked_indices]

        # Encode the unmasked tokens using the encoder's transformer
        encoded_tokens = self.encoder.transformer(tokens)
        
        masked_decoded_tokens = self.decode(encoded_tokens , masked_indices , unmasked_indices , 
                                            num_masked , device , batch_size, num_patches)
        
        # Reconstruct the pixel values from the masked decoded tokens
        pred_pixel_values = self.to_pixels(masked_decoded_tokens)
        
        pred_super_res = self.upsample(pred_pixel_values)
        #print(f"pred_pixel_values shape: {pred_pixel_values.shape}")
        recon_loss = self.compute_loss(pred_super_res, high_res_img)
        
        return recon_loss, pred_super_res, high_res_img

