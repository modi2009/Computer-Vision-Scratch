# Generative Adversarial Networks (GANs) - Portfolio

A portfolio showcasing my implementations of various GAN architectures in PyTorch, demonstrating progression from foundational concepts to advanced techniques in generative modeling.

## 🎯 Purpose

This repository demonstrates my understanding and implementation skills of different GAN architectures, showcasing:
- Deep learning fundamentals and PyTorch proficiency
- Progressive complexity from basic to advanced models
- Clean, documented, and production-ready code
- Understanding of state-of-the-art generative models

## 🗂️ Project Structure

```
GANs-Portfolio/
├── BasicGAN/          # Foundation: Vanilla GAN implementation
├── Pix2Pix/          # Conditional GANs for paired translation
├── CycleGAN/         # Unpaired image-to-image translation
└── ProGAN/           # Advanced: Progressive high-resolution generation
```

## 📚 Implemented Models

### 1. Basic GAN - Foundation
**What I Built**: A fully-connected GAN for MNIST digit generation

**Technical Implementation**:
- Custom discriminator and generator networks from scratch
- Binary Cross-Entropy loss implementation
- TensorBoard integration for training visualization
- Manual optimization with Adam (different learning rates for G/D)

**Key Learning**:
- GAN training dynamics and adversarial loss
- Balancing generator and discriminator updates
- Preventing mode collapse through LR tuning

**Code Highlights**:
```python
# Custom loss functions
def discriminator_loss(disc_real, disc_fake)
def generator_loss(disc_fake)

# Training loop with gradient management
loss_disc.backward(retain_graph=True)
opt_disc.step()
```

[📂 View Implementation](./BasicGAN/)

---

### 2. Pix2Pix - Conditional Translation
**What I Built**: Paired image-to-image translation with U-Net generator

**Technical Implementation**:
- **U-Net Generator**: Encoder-decoder with skip connections
- **PatchGAN Discriminator**: 70×70 receptive field patches
- Combined adversarial + L1 loss (weighted λ=100)
- PyTorch Lightning for scalable training
- Mixed precision (FP16) training
- Manual gradient accumulation

**Architecture Decisions**:
- Skip connections preserve spatial information
- PatchGAN focuses on high-frequency details
- L1 loss ensures structural similarity

**Code Highlights**:
```python
# U-Net with skip connections
skip_connections.append(x)
x = layer(torch.cat((x, skip_connections[i]), dim=1))

# Combined loss
L_G = BCE(D(G(x)), 1) + λ × L1(G(x), y)
```

[📂 View Implementation](./Pix2Pix/)

---

### 3. CycleGAN - Unpaired Translation
**What I Built**: Bidirectional translation without paired data (Horse ↔ Zebra)

**Technical Implementation**:
- Dual generators (G: X→Y, F: Y→X) and dual discriminators
- **Cycle consistency loss**: F(G(x)) ≈ x
- ResNet-based generator with 9 residual blocks
- Instance normalization for style transfer
- Identity loss for color preservation

**Novel Aspects**:
- No paired training data required
- Bidirectional consistency enforces structure
- Residual connections for complex transformations

**Code Highlights**:
```python
# Cycle consistency implementation
fake_zebra = gen_z(horse)
cycle_horse = gen_h(fake_zebra)
cycle_loss = L1(horse, cycle_horse) × λ_cycle

# Identity preservation
identity_horse = gen_h(horse)
identity_loss = L1(horse, identity_horse) × λ_identity
```

[📂 View Implementation](./CycleGAN/)

---

### 4. ProGAN - Progressive High-Resolution
**What I Built**: Progressive training for 1024×1024 face generation

**Technical Implementation**:
- **Progressive training**: 4×4 → 8×8 → ... → 1024×1024
- **Equalized learning rate**: Custom weight scaling per layer
- **Pixel normalization**: Prevents gradient explosion
- **Minibatch standard deviation**: Increases diversity
- WGAN-GP loss with gradient penalty
- Custom fade-in mechanism with alpha blending

**Advanced Techniques**:
```python
# Equalized learning rate (weight scaling)
class WSConv2d(nn.Module):
    scale = (gain / (in_channels * kernel_size²)) ** 0.5
    return conv(x * scale) + bias

# Pixel normalization
output = x / sqrt(mean(x²) + ε)

# Minibatch standard deviation
batch_stats = std(x, dim=0).mean().repeat(batch_size)
output = concat([x, batch_stats])
```

**Progressive Fade-in**:
```python
def fade(self, alpha, upscaled, generated):
    return alpha * generated + (1 - alpha) * upscaled
```

[📂 View Implementation](./ProGAN/)

## 💡 Technical Skills Demonstrated

### Deep Learning Fundamentals
- ✅ Custom loss function design
- ✅ Adversarial training dynamics
- ✅ Gradient flow management
- ✅ Normalization techniques (Batch, Instance, Pixel)
- ✅ Regularization (Dropout, Gradient Penalty)

### PyTorch Proficiency
- ✅ Custom `nn.Module` implementations
- ✅ Manual optimization and backward passes
- ✅ DataLoader and Dataset customization
- ✅ Mixed precision training
- ✅ PyTorch Lightning integration
- ✅ TensorBoard logging

### Advanced Architectures
- ✅ U-Net with skip connections
- ✅ ResNet with residual blocks
- ✅ Progressive neural networks
- ✅ PatchGAN discriminators
- ✅ Equalized learning rate implementation

### Training Techniques
- ✅ Gradient accumulation
- ✅ Learning rate scheduling
- ✅ Gradient penalty (WGAN-GP)
- ✅ Cycle consistency
- ✅ Progressive training
- ✅ Alpha fade-in mechanism

## 📊 Complexity Progression

| Model | Architecture Complexity | Training Technique | Innovation Level |
|-------|------------------------|-------------------|------------------|
| Basic GAN | Simple FC layers | Standard adversarial | Foundation |
| Pix2Pix | U-Net + PatchGAN | Paired + L1 loss | Conditional generation |
| CycleGAN | Dual ResNet + Dual Disc | Cycle consistency | Unpaired translation |
| ProGAN | Progressive CNN | Multi-stage + weight scaling | High-res generation |

## 🔬 Implementation Highlights

### Custom Components Built

1. **Weight-Scaled Convolutions** (ProGAN)
   - Equalized learning rate across layers
   - Custom initialization and forward pass

2. **Pixel Normalization** (ProGAN)
   - Feature normalization to unit sphere
   - Prevents exploding gradients

3. **Cycle Consistency** (CycleGAN)
   - Bidirectional translation enforcement
   - Multiple loss components coordination

4. **Progressive Training Loop** (ProGAN)
   - Dynamic resolution scaling
   - Alpha-based fade-in mechanism
   - Adaptive batch sizing

5. **U-Net Architecture** (Pix2Pix)
   - Encoder-decoder with skip connections
   - Feature concatenation handling

## 📈 Results & Capabilities

### Basic GAN
- ✅ Generated realistic MNIST digits
- ✅ Implemented from-scratch GAN training
- ✅ Learned adversarial dynamics

### Pix2Pix
- ✅ Satellite → Map translation
- ✅ Structural preservation with L1 loss
- ✅ High-frequency detail with PatchGAN

### CycleGAN
- ✅ Horse ↔ Zebra bidirectional translation
- ✅ No paired data required
- ✅ Identity and cycle consistency

### ProGAN
- ✅ Progressive training 4×4 to 1024×1024
- ✅ High-quality face generation
- ✅ Stable training with advanced techniques

## 🛠️ Code Quality Features

- **Modular Design**: Reusable components and clear separation
- **Documentation**: Comprehensive README for each model
- **Type Hints**: Clear function signatures (where applicable)
- **Comments**: Explaining complex operations
- **Error Handling**: Robust data loading and training
- **Logging**: TensorBoard and progress tracking
- **Reproducibility**: Fixed seeds and hyperparameters

## 🧠 Key Insights Gained

### Training Dynamics
- Balancing discriminator and generator strength
- Learning rate ratio importance (e.g., 1e-3 vs 3e-7 in Basic GAN)
- Gradient penalty for Lipschitz constraint in WGAN

### Architecture Choices
- Skip connections preserve information in U-Net
- Residual blocks enable deeper networks in CycleGAN
- Instance norm works better than batch norm for style transfer

### Loss Design
- L1 loss captures structure better than L2
- Cycle consistency prevents arbitrary mappings
- Gradient penalty stabilizes Wasserstein distance

### Advanced Techniques
- Progressive training enables high resolution
- Weight scaling equalizes learning across layers
- Minibatch statistics increase output diversity

## 📚 Technologies Used

- **Framework**: PyTorch 1.12+, PyTorch Lightning
- **Training**: CUDA, Mixed Precision (FP16)
- **Data**: Albumentations, torchvision transforms
- **Visualization**: TensorBoard, matplotlib
- **Progress**: tqdm
- **Environment**: Kaggle Notebooks, Google Colab

## 🎓 Academic References

All implementations based on seminal papers:

1. **Goodfellow et al. (2014)** - Generative Adversarial Networks
2. **Isola et al. (2017)** - Image-to-Image Translation with Conditional Adversarial Networks
3. **Zhu et al. (2017)** - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
4. **Karras et al. (2017)** - Progressive Growing of GANs for Improved Quality, Stability, and Variation

## 📊 Model Comparison

| Feature | Basic GAN | Pix2Pix | CycleGAN | ProGAN |
|---------|-----------|---------|----------|---------|
| **Loss Type** | BCE | BCE + L1 | BCE + Cycle + Identity | Wasserstein + GP |
| **Normalization** | None | Batch | Instance | Pixel |
| **Architecture** | FC | CNN (U-Net) | ResNet | Progressive CNN |
| **Key Innovation** | Adversarial | Paired translation | Unpaired translation | Progressive training |
| **Code Lines** | ~150 | ~400 | ~450 | ~500 |
| **Training Complexity** | Low | Medium | High | Very High |

## 💻 Development Process

### Approach
1. **Paper Study**: Deep understanding of architecture and loss
2. **Modular Design**: Build reusable components first
3. **Incremental Testing**: Test each component individually
4. **Training Experiments**: Tune hyperparameters systematically
5. **Documentation**: Write comprehensive READMEs

### Challenges Overcome
- **Mode Collapse**: Tuned learning rates and added regularization
- **Memory Management**: Implemented gradient accumulation and mixed precision
- **Training Instability**: Applied gradient penalty and normalization
- **Progressive Fade-in**: Implemented smooth alpha blending mechanism

## 🚀 What This Demonstrates

This portfolio showcases my ability to:

✅ **Understand Complex Papers**: Implement models from academic papers  
✅ **Write Production Code**: Clean, documented, maintainable implementations  
✅ **Deep Learning Expertise**: Advanced architectures and training techniques  
✅ **Problem Solving**: Debug training issues and optimize performance  
✅ **PyTorch Mastery**: Custom modules, optimization, and training loops  
✅ **Research to Code**: Translate mathematical formulations to working code  

## 📝 Notes

- Each folder contains detailed implementation documentation
- Code emphasizes clarity and educational value over brevity
- Training logs and sample outputs demonstrate successful convergence
- Implementations prioritize understanding over using pre-built libraries

---

**Portfolio Purpose**: Demonstrating deep learning expertise and PyTorch proficiency through progressive GAN implementations

*Developed as part of computer vision and deep learning learning journey*