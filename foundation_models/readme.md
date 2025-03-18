## Specific Test VI. Foundation Model 

### Task VI.A: 
* Train a ```Masked Autoencoder (MAE)``` on the no_sub samples from the provided dataset to learn a feature representation of strong lensing images. The MAE should be trained for reconstructing masked portions of input images. Once this pre-training phase is complete, fine-tune the model on the full dataset for a multi-class classification task to distinguish between the three classes. Please implement your approach in PyTorch or Keras and discuss your strategy.
* ```Dataset```: https://drive.google.com/file/d/1znqUeFzYz-DeAE3dYXD17qoMPK82Whji/view?usp=sharing
* ```Dataset Description```: The Dataset consists of three classes: no_sub (no substructure), cdm (cold dark matter substructure), and axion (axion-like particle substructure).
* ```Evaluation Metrics```: ROC curve (Receiver Operating Characteristic curve) and AUC score (Area Under the ROC Curve) 

### Task VI.B: 
* Take the pre-trained model from Task VI.A and ```fine-tune it for a super-resolution task```. The model should be fine-tuned to upscale low-resolution strong lensing images using the provided high-resolution samples as ground truths. Please implement your approach in PyTorch or Keras and discuss your strategy.
* ```Dataset```: https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view?usp=sharing
* ```Dataset Description```: The dataset comprises simulated strong lensing images with no substructure at multiple resolutions: high-resolution (HR) and low-resolution (LR).
* ```Evaluation Metrics```: MSE (Mean Squared Error), SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio)