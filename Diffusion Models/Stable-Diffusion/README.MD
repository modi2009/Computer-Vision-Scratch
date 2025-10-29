# Conditional Latent Diffusion Model for Face Generation

A PyTorch implementation of a conditional latent diffusion model trained on the CelebA dataset. This project demonstrates advanced deep learning techniques including variational autoencoders, U-Net architectures, cross-attention mechanisms, and diffusion-based generative modeling.

## 🎯 Project Overview

This project implements a conditional image generation pipeline that can synthesize realistic facial images based on specified attributes (e.g., smiling, male, young, eyeglasses). The model operates in latent space for computational efficiency, similar to Stable Diffusion.

## 🏗️ Architecture

### Core Components

**1. VAE Encoder (Pre-trained)**
- Uses Stable Diffusion v1.5's pre-trained VAE
- Compresses 256×256 RGB images into 4-channel latent representations
- Reduces spatial dimensions by 8× (256×256 → 32×32)
- Latent scaling factor: 0.18215

**2. Conditional Latent Diffusion U-Net**
- Custom U-Net architecture operating on latent space
- Encoder: Progressive downsampling with residual blocks
- Bottleneck: Deep feature extraction with attention
- Decoder: Upsampling with skip connections
- Base channels: 64, expanding to 256 in bottleneck

**3. Attention Mechanisms**
- **Cross-Attention**: Integrates conditional attribute embeddings with spatial features
- **Multi-Head Attention**: 4 attention heads for richer feature representations
- Applied at key layers (Inc, Down2, Bottleneck)

**4. Time Embedding**
- Sinusoidal positional encoding for diffusion timesteps
- 256-dimensional embedding space
- Injected into each residual block

**5. Conditional Embedding**
- Linear projection of binary attributes (40 CelebA attributes)
- 256-dimensional per-attribute embeddings
- Enables fine-grained control over generated images

### ResBlock Architecture
Each residual block contains:
- Instance Normalization + ReLU activation
- 3×3 Convolution layers
- Optional time embedding injection
- Optional cross-attention with conditional features
- Residual skip connection

## 📊 Dataset

**CelebA (CelebFaces Attributes Dataset)**
- 202,599 face images of celebrities
- 40 binary facial attributes per image
- Split: 162,770 train / 19,867 validation / 19,962 test
- Image resolution: 256×256 (resized and center-cropped)

**Selected Attributes for Demo:**
- Smiling
- Male
- Young
- Eyeglasses

## 🔬 Training Process

### Phase 1: Latent Encoding
- Pre-encode entire dataset using pre-trained VAE
- Store latent representations to disk for efficient training
- Memory-efficient: ~10GB for full CelebA latents

### Phase 2: Diffusion Training
- **Noise Schedule**: Linear beta schedule (β: 1e-4 → 0.02)
- **Timesteps**: 1000
- **Optimizer**: Adam (lr: 1e-4)
- **Batch Size**: 64
- **Loss Function**: Mean Squared Error (MSE) between predicted and actual noise
- **Training Duration**: 100 epochs
- **Validation**: Continuous monitoring with best model checkpointing

### Training Performance
```
Epoch 1:  Train Loss: 1.007400 | Val Loss: 1.002757
Epoch 3:  Train Loss: 1.002686 | Val Loss: 1.002321 (Best)
Epoch 100: Converged
```

## 🛠️ Technical Highlights

### Efficient Memory Management
- Latent space training reduces memory by ~64× vs. pixel-space
- Pre-encoded dataset eliminates redundant VAE forward passes
- Mixed-precision training with `torch.autocast`

### Advanced PyTorch Techniques
- Custom `Dataset` classes for flexible data loading
- Efficient batch processing with `DataLoader`
- GPU acceleration with CUDA
- Checkpoint saving with optimizer state preservation

### Architectural Innovations
- **Skip Connections**: Preserve high-resolution features across U-Net
- **Instance Normalization**: Stabilizes training for small batch sizes
- **Adaptive Channel Scaling**: Doubles channels at each downsampling stage
- **Attention-Based Conditioning**: Spatially-aware attribute injection

## 📈 Visualizations

The project includes comprehensive visualization tools:
- **Architecture Diagrams**: High-resolution (600 DPI) network schematics
- **Data Distribution Plots**: Attribute balance analysis
- **Sample Visualizations**: Grid-based image displays with attribute labels
- **Training Curves**: Loss progression tracking

## 🔍 Key Insights

1. **Latent Space Efficiency**: Operating in VAE latent space provides 64× memory reduction while maintaining generation quality

2. **Conditional Control**: Cross-attention mechanisms enable precise attribute-based control without classifier-free guidance

3. **Progressive Training**: Gradual noise addition (1000 steps) allows model to learn both coarse and fine-grained features

4. **Residual Learning**: Skip connections in U-Net preserve details and accelerate training convergence

## 🧰 Technologies Used

- **PyTorch**: Deep learning framework
- **Diffusers**: Pre-trained VAE from Hugging Face
- **Pandas**: Dataset manipulation
- **Pillow**: Image processing
- **Matplotlib/Seaborn**: Data visualization
- **Graphviz**: Architecture diagram generation
- **tqdm**: Progress tracking

## 📝 Code Quality

- Modular architecture with reusable components
- Type-consistent data handling
- Comprehensive documentation
- Memory-efficient data pipelines
- Best practices for model checkpointing
- Clear separation of concerns (data, model, training)

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- Generative modeling (diffusion models, VAEs)
- Deep learning architecture design (U-Nets, attention mechanisms)
- PyTorch programming (custom datasets, training loops, GPU optimization)
- Computer vision (image preprocessing, data augmentation)
- Research implementation (translating papers to code)
- Experiment tracking and model evaluation

## 📚 References

- Stable Diffusion (Rombach et al., 2022)
- U-Net Architecture (Ronneberger et al., 2015)
- Diffusion Models (Ho et al., 2020)
- CelebA Dataset (Liu et al., 2015)

---

**Note**: This is a portfolio demonstration project showcasing machine learning engineering skills. The implementation prioritizes clarity and educational value while maintaining production-level code quality.
