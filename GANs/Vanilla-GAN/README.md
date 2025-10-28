# GAN for MNIST Digit Generation

A simple PyTorch implementation of Generative Adversarial Networks (GANs) for generating handwritten digits using the MNIST dataset.

## Overview

This project implements a basic GAN with fully connected layers to generate realistic handwritten digits. The model learns to create new digit images by training two networks in an adversarial manner: a Generator that creates fake images and a Discriminator that tries to distinguish real from fake images.

## Architecture

### Generator
- **Input**: Random noise vector (64 dimensions)
- **Hidden Layer**: 256 neurons with LeakyReLU activation
- **Output**: 784 neurons (28×28 flattened image) with Tanh activation
- Transforms random noise into realistic digit images

### Discriminator
- **Input**: Flattened image (784 dimensions)
- **Hidden Layer**: 128 neurons with LeakyReLU activation
- **Output**: Single neuron with Sigmoid activation (real/fake probability)
- Binary classifier for authenticity verification

## Requirements

```python
torch
torchvision
matplotlib
tensorboard
tqdm
```

Install dependencies:
```bash
pip install torch torchvision matplotlib tensorboard tqdm
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Dimensions | 784 (28×28) | Flattened MNIST image size |
| Noise Dimension | 64 | Random input vector size |
| Batch Size | 32 | Samples per training iteration |
| Epochs | 50 | Total training iterations |
| Generator LR | 1e-3 | Learning rate for generator |
| Discriminator LR | 3e-7 | Learning rate for discriminator |
| Optimizer | Adam | β1=0.5, β2=0.999 |

## Loss Functions

### Binary Cross-Entropy Loss (BCE)

**Discriminator Loss:**
```
L_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0)
```
- Maximizes ability to classify real images as real and fake as fake

**Generator Loss:**
```
L_G = BCE(D(G(z)), 1)
```
- Minimizes discriminator's ability to detect fake images

## Dataset

**MNIST Dataset:**
- 60,000 training images of handwritten digits (0-9)
- 28×28 grayscale images
- Normalized to [-1, 1] range using `transforms.Normalize((0.5,), (0.5,))`
- Automatically downloaded if not present

## Training Process

1. **Discriminator Training:**
   - Generate fake images from random noise
   - Pass real images through discriminator
   - Pass fake images through discriminator (detached)
   - Compute combined loss and update discriminator weights

2. **Generator Training:**
   - Generate fake images from noise
   - Pass through discriminator (no detach)
   - Compute loss based on discriminator's output
   - Update generator weights

3. **Monitoring:**
   - Real-time progress bars with tqdm
   - TensorBoard logging for visual monitoring
   - Saves generated images at each epoch

## Usage

### Basic Training

```python
# Run the entire script
python gan_mnist.py
```

### TensorBoard Visualization

```bash
# Launch TensorBoard
tensorboard --logdir runs/GAN_MNIST

# Or in Jupyter/Colab
%load_ext tensorboard
%tensorboard --logdir runs/GAN_MNIST
```

### Loading Saved Models

```python
# Load generator
gen = Generator(z_dim=64, img_dim=784)
gen.load_state_dict(torch.load('generator_epoch_40.pth'))
gen.eval()

# Generate new images
noise = torch.randn(16, 64)
fake_images = gen(noise).reshape(-1, 1, 28, 28)
```

## Output Structure

```
project/
├── runs/
│   └── GAN_MNIST/
│       ├── fake/          # TensorBoard logs for generated images
│       └── real/          # TensorBoard logs for real images
├── dataset/               # MNIST dataset (auto-downloaded)
├── generator_epoch_*.pth  # Saved generator models
└── discriminator_epoch_*.pth  # Saved discriminator models
```

## Training Features

- **Progress Tracking**: Real-time loss monitoring with tqdm
- **TensorBoard Integration**: Visual monitoring of training progress
- **Model Checkpointing**: Saves models every 10 epochs
- **Fixed Noise**: Consistent noise vector for tracking generator improvement
- **Gradient Retention**: Proper gradient flow management

## Key Training Details

### Learning Rate Balance
- **Discriminator LR (3e-7)**: Very low to prevent overpowering the generator
- **Generator LR (1e-3)**: Higher to help catch up with discriminator
- This imbalance helps stabilize GAN training

### Gradient Management
- Uses `retain_graph=True` during discriminator backpropagation
- Detaches fake images when training discriminator to prevent generator updates
- Does not detach when training generator to allow gradient flow

## Training Output

Each epoch displays:
```
Epoch [1/50] | Avg D Loss: 0.7234 | Avg G Loss: 1.2456
```

Progress bars show:
- Current batch/total batches
- Instantaneous discriminator and generator losses
- Processing speed

## Monitoring Training

### Signs of Good Training:
- Discriminator loss stabilizes around 0.5-0.7
- Generator loss gradually decreases
- Generated images become recognizable digits over time

### Common Issues:
- **Mode Collapse**: Generator produces limited variety → restart with different initialization
- **Discriminator Too Strong**: D_loss near 0 → increase generator LR or decrease discriminator LR
- **Generator Too Strong**: G_loss near 0, D_loss increases → opposite adjustment

## Visualization

Generated images are logged to TensorBoard:
- **Fake Images**: Track generator progress over epochs
- **Real Images**: Reference for comparison
- Grid layout shows multiple samples simultaneously

## Customization

### Change Image Resolution
```python
image_dim = 32 * 32 * 1  # For 32×32 images
```

### Adjust Architecture
```python
# Add more layers or neurons
self.gen = nn.Sequential(
    nn.Linear(z_dim, 512),
    nn.LeakyReLU(0.1),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.1),
    nn.Linear(256, img_dim),
    nn.Tanh(),
)
```

### Different Dataset
```python
dataset = datasets.FashionMNIST(root="dataset/", transform=transform, download=True)
```

## References

- [Original GAN Paper](https://arxiv.org/abs/1406.2661): Generative Adversarial Networks by Ian Goodfellow et al.
- [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## Tips for Better Results

1. **Train Longer**: 50 epochs might not be enough for high-quality results
2. **Balanced Training**: Monitor loss ratios to ensure neither network dominates
3. **Experiment with Learning Rates**: The balance is crucial for stable training
4. **Use DCGAN**: For better image quality, consider convolutional architectures
5. **Batch Normalization**: Can improve training stability

## License

This implementation is for educational and research purposes.