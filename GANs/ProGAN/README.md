# ProGAN: Progressive Growing of GANs

A PyTorch implementation of Progressive Growing of GANs (ProGAN) for high-resolution image generation using the CelebA-HQ dataset.

## Overview

ProGAN progressively trains a GAN by starting with low-resolution images (4×4) and gradually increasing the resolution by adding new layers. This approach enables stable training of GANs that can generate high-resolution images up to 1024×1024 pixels.

## Key Innovations

- **Progressive Training**: Incrementally grows both generator and discriminator
- **Smooth Fading**: New layers are smoothly faded in using alpha blending
- **Equalized Learning Rate**: Weight scaling for stable training
- **Pixel Normalization**: Prevents gradient explosion in the generator
- **Minibatch Standard Deviation**: Increases variety in generated images
- **Wasserstein Loss with Gradient Penalty**: Stable adversarial training

## Architecture

### Generator
- **Progressive Structure**: Starts at 4×4, grows to 512×1024
- **Initial Block**: 
  - ConvTranspose2d (Z_dim → in_channels) 4×4
  - WSConv2d with PixelNorm
- **Progressive Blocks**: 
  - Upsampling (2×) + Conv2dBlock for each resolution
  - Each block: WSConv2d → PixelNorm → LeakyReLU
- **RGB Layers**: Separate 1×1 conv for each resolution
- **Fade-in**: Smoothly blends upscaled previous output with new output

### Discriminator (Critic)
- **Progressive Structure**: Mirror of generator
- **Progressive Blocks**: Conv2dBlock + AvgPool2d
- **RGB Layers**: 1×1 conv from RGB to features at each resolution
- **Minibatch StdDev**: Concatenates batch statistics before final block
- **Final Block**: 
  - WSConv2d layers
  - Flatten + Linear → single scalar output

### Custom Components

#### Equalized Learning Rate (Weight Scaling)
```python
scale = sqrt(2 / (in_channels * kernel_size²))
output = conv(input * scale) + bias
```

#### Pixel Normalization
```python
output = x / sqrt(mean(x²) + ε)
```

#### Minibatch Standard Deviation
```python
batch_std = std(x, dim=0).mean()
output = concat([x, batch_std.repeat()])
```

## Training Schedule

| Step | Resolution | Batch Size | Epochs |
|------|------------|------------|--------|
| 0 | 4×4 | 16 | 10 |
| 1 | 8×8 | 16 | 10 |
| 2 | 16×16 | 16 | 10 |
| 3 | 32×32 | 16 | 10 |
| 4 | 64×64 | 16 | 10 |
| 5 | 128×128 | 16 | 10 |
| 6 | 256×256 | 16 | 10 |
| 7 | 512×512 | 8 | 10 |
| 8 | 1024×1024 | 4 | 10 |

**Total Training Time**: ~90 epochs across all resolutions

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Z Dimension | 256 | Latent noise vector size |
| In Channels | 256 | Base feature map size |
| Learning Rate | 1e-3 | For both G and C |
| λ_GP | 10 | Gradient penalty weight |
| Image Channels | 3 | RGB output |
| Target Resolution | 512×512 | Maximum image size |
| Optimizer | Adam | β1=0.0, β2=0.99 |
| Device | GPU (CUDA) | Required for training |

## Requirements

```python
torch >= 1.12.0
torchvision
numpy
PIL
albumentations
tqdm
```

Install dependencies:
```bash
pip install torch torchvision numpy pillow albumentations tqdm
```

## Dataset Structure

```
celeba_hq/
└── train/
    ├── class_folder/  # ImageFolder format
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
```

**CelebA-HQ Dataset**: 30,000 high-quality celebrity face images at 1024×1024 resolution

## Loss Functions

### Wasserstein Loss with Gradient Penalty (WGAN-GP)

**Critic Loss:**
```
L_C = -[E(C(x_real)) - E(C(G(z)))] + λ_GP × GP + 0.001 × E(C(x_real)²)
```
- Wasserstein distance (maximized by critic)
- Gradient penalty for Lipschitz constraint
- Drift penalty (0.001 term) prevents unbounded critic values

**Generator Loss:**
```
L_G = -E(C(G(z)))
```
- Maximizes critic score on fake images

### Gradient Penalty
```python
GP = E[(||∇_x̂ C(x̂)||₂ - 1)²]
where x̂ = β × x_real + (1-β) × x_fake
```

## Progressive Training Mechanics

### Alpha Fade-in Schedule
```python
alpha = 0.00001  # Start
alpha += batch_size / (dataset_size × epochs × 0.5)  # Increment
alpha = min(alpha, 1.0)  # Cap at 1.0
```

### Generator Fade
```python
output = tanh(α × new_layer + (1-α) × upscaled_old)
```

### Discriminator Fade
```python
output = α × new_path + (1-α) × downscaled_old
```

## Data Augmentation

```python
transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

## Usage

### Basic Training

```python
# Initialize models
gen = Generator(z_dim=256, in_channels=256, image_channels=3)
critic = Discriminator(in_channels=256, image_channels=3)

# Initialize optimizers
opt_gen = optim.Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.99))

# Progressive epochs for each resolution
progressive_epochs = [10, 10, 10, 10, 10, 10, 10, 10, 10]
batch_sizes = [16, 16, 16, 16, 16, 16, 16, 8, 4]

# Train
train(progressive_epochs, critic, gen, opt_critic, opt_gen, batch_sizes, device)
```

### Generate Images

```python
gen.eval()
with torch.no_grad():
    noise = torch.randn(8, 256, 1, 1).to(device)
    # Generate at specific resolution
    step = 6  # 256×256 resolution
    alpha = 1.0  # Fully faded in
    fake_images = gen(noise, alpha, step)
```

## Training Process

For each resolution step:

1. **Initialize Alpha**: Start with α = 0.00001
2. **Load Data**: Create dataloader for current resolution
3. **Train Epoch**:
   - **Critic Training**:
     - Generate fake images
     - Compute real and fake critic scores
     - Calculate gradient penalty
     - Update critic
   - **Generator Training**:
     - Generate fake images
     - Compute critic score
     - Update generator
   - **Update Alpha**: Gradually increase toward 1.0
4. **Save Progress**: Generate sample image after each epoch
5. **Next Resolution**: Move to next step

## Output Structure

```
project/
├── images/
│   ├── generated_image0.png  # 4×4
│   ├── generated_image1.png  # 8×8
│   ├── generated_image2.png  # 16×16
│   ├── generated_image3.png  # 32×32
│   ├── generated_image4.png  # 64×64
│   ├── generated_image5.png  # 128×128
│   ├── generated_image6.png  # 256×256
│   ├── generated_image7.png  # 512×512
│   └── generated_image8.png  # 1024×1024
```

## Training Features

- **Progressive Resolution**: Smooth transition between resolutions
- **Fixed Noise**: Consistent latent vector tracks progress
- **Real-time Monitoring**: tqdm progress bars with loss values
- **Automatic Image Saving**: Visual progress tracking
- **Dynamic Batch Sizes**: Adjusts for GPU memory constraints
- **Stable Training**: Multiple techniques prevent mode collapse

## Key Training Details

### Factors Array
```python
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
```
Controls channel scaling at each resolution for efficient computation.

### Resolution Progression
```python
image_size = 4 × 2^step
# step 0: 4×4
# step 1: 8×8
# step 2: 16×16
# ...
# step 8: 1024×1024
```

### Batch Size Reduction
Larger images require smaller batches to fit in GPU memory:
- 4×4 to 256×256: batch_size = 16
- 512×512: batch_size = 8
- 1024×1024: batch_size = 4

## Monitoring Training

### Expected Loss Patterns

**Critic Loss**: Negative values (Wasserstein distance)
```
Early training: -0.5 to -1.0
Stable training: -0.1 to -0.3
```

**Generator Loss**: Negative values (adversarial objective)
```
Early training: 0.5 to 1.5
Stable training: 0.1 to 0.3
```

### Visual Quality Progression
- **Step 0-2 (4×4 to 16×16)**: Blurry, color blobs
- **Step 3-4 (32×32 to 64×64)**: Face structure emerges
- **Step 5-6 (128×128 to 256×256)**: Clear facial features
- **Step 7-8 (512×512 to 1024×1024)**: High-quality, detailed faces

## Advantages of ProGAN

1. **Stable Training**: Progressive approach reduces instability
2. **High Resolution**: Can generate 1024×1024 images
3. **Fast Convergence**: Each stage trains faster than full resolution
4. **Memory Efficient**: Smaller images in early stages
5. **Quality Control**: Can stop at any resolution if quality degrades

## Common Issues & Solutions

### Mode Collapse
**Symptoms**: Generator produces limited variety
**Solutions**: 
- Verify minibatch stddev is working
- Check gradient penalty calculation
- Ensure alpha fade is progressing correctly

### Training Instability
**Symptoms**: Loss values spike or diverge
**Solutions**:
- Verify weight scaling implementation
- Check gradient penalty weight (try λ=5 or λ=15)
- Reduce learning rate to 5e-4

### Poor Image Quality
**Symptoms**: Blurry or artifact-heavy images
**Solutions**:
- Train longer at each resolution (15-20 epochs)
- Verify pixel normalization in generator
- Check data normalization [-1, 1] range

### GPU Out of Memory
**Solutions**:
- Reduce batch sizes further
- Lower maximum resolution to 512×512
- Use gradient checkpointing
- Enable mixed precision training

## Customization

### Change Target Resolution
```python
IMAGE_SIZE = 256  # Instead of 512
NUM_STEPS = int(log2(IMAGE_SIZE/4)) + 1  # Recalculate steps
```

### Modify Training Schedule
```python
# Longer training at each resolution
PROGRESSIVE_EPOCHS = [20, 20, 20, 15, 15, 15, 10, 10, 10]

# Different batch sizes
BATCH_SIZES = [32, 32, 32, 16, 16, 8, 4, 2, 1]
```

### Adjust Architecture Capacity
```python
# More capacity for higher quality
IN_CHANNELS = 512
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
```

## Performance Notes

- **Training Time**: 
  - 4×4 to 64×64: ~30 minutes per resolution
  - 128×128: ~1 hour
  - 256×256: ~2 hours
  - 512×512+: ~4-8 hours per resolution
- **GPU Memory**: 
  - 12GB+ VRAM recommended
  - Can train up to 256×256 with 8GB VRAM
- **Total Training**: 24-48 hours for full 512×512 training

## References

- [ProGAN Paper](https://arxiv.org/abs/1710.10196): Progressive Growing of GANs for Improved Quality, Stability, and Variation
- [Original Implementation](https://github.com/tkarras/progressive_growing_of_gans): Official TensorFlow code
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028): Improved Training of Wasserstein GANs

## Best Practices

1. **Monitor Alpha Values**: Ensure smooth fade-in progression
2. **Check Generated Samples**: Visual quality is best metric
3. **Save Checkpoints**: Save models at each resolution milestone
4. **Use Fixed Noise**: Consistent latent vector shows progress clearly
5. **Balanced Training**: Neither critic nor generator should dominate
6. **Patience**: High-resolution training takes time

## Advanced Features

### Equalized Learning Rate Benefits
- Prevents certain layers from dominating training
- Ensures all layers learn at similar rates
- More stable than traditional initialization

### Pixel Normalization Benefits
- Prevents signal explosion in generator
- Normalizes features to unit sphere
- Helps training stability

### Minibatch StdDev Benefits
- Encourages diversity in generated images
- Provides critic with batch-level statistics
- Reduces mode collapse

## Tips for Best Results

1. **Dataset Quality**: High-quality, aligned faces work best
2. **Data Augmentation**: Random flips increase variety
3. **Learning Rate**: Start at 1e-3, reduce if unstable
4. **Gradient Penalty**: Keep λ_GP = 10 for most cases
5. **Progressive Schedule**: Don't rush—allow proper fade-in
6. **Hardware**: Use GPU with at least 8GB VRAM

## License

This implementation is for educational and research purposes.

## Acknowledgments

- Based on the ProGAN paper by Karras et al. (NVIDIA)
- CelebA-HQ dataset for high-quality face images
- PyTorch community for framework support