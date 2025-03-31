# ✨ DeepLense Submission Proposal - 2025 ✨

### 📜 Submission Instructions:

```
- You are required to submit Jupyter Notebooks for each task clearly showing your implementation.
- Please also put your solution in a github repository and send us a link
- You must calculate and present the required evaluation metrics for the validation data (90:10 train-test split).
- Submission deadline is April 1
```

## 🔭 Project of Interest:

- My primary focus lies in **Diffusion Models**. I've been deeply engaged with the advancements in this exciting area of generative AI.
- I possess a strong theoretical and mathematical understanding, and I have implemented key diffusion model architectures such as **DDPM**, **DDIM**, and **Latent Diffusion Models**.
- As an optional third task, I explored **Foundation Models**. This was a new and enriching experience, providing valuable insights into large-scale pre-training and the adaptation of learned representations for diverse downstream tasks like classification and super-resolution.

---


**Slurm scripts** for managing batch processes are in the ```scripts/``` directory 📂. Utilizing my institute's GPU clusters 🚀, I explored distributed training by running Python scripts (.py) via SSH. **While I apologize 🙏 for not providing Jupyter Notebooks directly**, please be assured that I've conducted thorough **quantitative and qualitative analyses** 📊🔍 for each task. The **dataset split logic** (90:10 ratio) is implemented **[here](data_processing.py)** ⚙️.

---

## 📊 1. Common Multi-class Classification:

- I fine-tuned a **Resnet-18** model (pretrained on ImageNet) to classify gravitational lensing images into the following categories:
    - **No Substructure**
    - **️Subhalo Substructure**
    - **Vortex Substructure**
- Model performance was rigorously evaluated using key metrics: **Accuracy**, **Kappa**, **ROC-AUC curve**, and the **MSE loss curve**.
- A comprehensive description of the training setup and a detailed analysis of the results can be found **[here](Multi_Class_Classification/ReadMe.md)**.

## 🌌 2. Specific Test IV. Diffusion Models:

- I successfully trained a **DDIM model** to generate realistic strong gravitational lensing images.
- To accommodate larger batch sizes, the model training was distributed across **2 GPUs**.
- Detailed information regarding the training setup and a thorough analysis of the generated images are available **[here](diffusion/readme.md)**.
- The foundational architecture code for the DDIM/DDPM implementation draws inspiration from this excellent **[open-source repository](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main)**.

## 🧠 3. Specific Test VI. Foundation Model:

- I undertook the pre-training of a **ViT** as an encoder, coupled with a lightweight decoder, utilizing the **Masked AutoEncoder (MAE)** technique.
- **Task VI-A (Classification):** I employed the pre-trained ViT with an MLP head as the encoder for classification tasks.
- **Task VI-B (Super-Resolution):** I integrated the pre-trained ViT with a more powerful decoder to perform super-resolution.
- A detailed account of the training procedures and a comprehensive analysis of the outcomes for both sub-tasks are documented **[here](foundation_models/readme.md)**.
- This insightful **[blog post](https://towardsdatascience.com/how-to-implement-state-of-the-art-masked-autoencoders-mae-6f454b736087/)** proved invaluable for understanding and implementing the Masked AutoEncoder.


---
### ✨ It was a truly engaging project, and I'm delighted to have had such an exciting and insightful learning experience. Thank you for the opportunity! ✨
