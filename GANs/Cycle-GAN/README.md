# CycleGAN: Horse to Zebra Image Translation

A PyTorch implementation of CycleGAN for unpaired image-to-image translation between horses and zebras.

## Overview

CycleGAN is a method for training image-to-image translation models without paired examples. This implementation converts horses to zebras and vice versa using cycle consistency loss.

## Architecture

### Generator
- Encoder-decoder architecture with residual blocks
- Initial 7×7 convolution layer
- 2 downsampling layers (stride 2)
- 9 residual blocks for transformation
- 2 upsampling layers (transposed convolution)
- Final 7×7 convolution with Tanh activation
- Instance normalization throughout

### Discriminator
- PatchGAN discriminator (30×30 output)
- 4 convolutional layers with LeakyReLU
- Instance normalization
- Sigmoid output for real/fake classification

## Loss Functions

1. **Adversarial Loss**: MSE loss for discriminator predictions
2. **Cycle Consistency Loss**: L1 loss ensuring `F(G(x)) ≈ x`
3. **Identity Loss**: Optional L1 loss for color preservation

Total Generator Loss = Adversarial + λ_cycle × Cycle + λ_identity × Identity

## Requirements

```python
torch
torchvision
numpy
albumentations
PIL
tqdm
```

## Dataset Structure

```
horse2zebra/
├── trainA/  # Horse images
├── trainB/  # Zebra images
├── testA/   # Test horse images
└── testB/   # Test zebra images
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 256×256 |
| Batch Size | 4 |
| Learning Rate | 2e-4 |
| Epochs | 200 |
| λ_cycle | 10.0 |
| λ_identity | 0.0 |
| Optimizer | Adam (β1=0.5, β2=0.999) |

## Training

The model alternates between training:
1. **Discriminators** (D_H and D_Z): Distinguish real from generated images
2. **Generators** (G_H and G_Z): Generate realistic images that fool discriminators

Training includes:
- Cycle consistency: Horse → Zebra → Horse
- Identity mapping: Zebra → G_Z → Zebra (optional)
- Adversarial training for both directions

## Data Augmentation

Using Albumentations:
- Resize to 256×256
- Normalize to [-1, 1] range
- Convert to PyTorch tensors

## Output

Generated images are saved every 200 batches during training:
- `horse_{idx}.png`: Generated horse images
- `zebra_{idx}.png`: Generated zebra images

## Model Components

### Key Functions
- `Conv2D()`: Convolutional block with InstanceNorm and LeakyReLU
- `Conv2DGen()`: Generator conv block with ReLU
- `ConvTranspose2D()`: Upsampling block
- `ResidualBlock()`: Skip connection for feature preservation
- `discriminator_loss()`: MSE-based adversarial loss
- `generator_loss()`: Combined adversarial, cycle, and identity loss

## Training Process

```python
for epoch in range(200):
    for horse, zebra in dataloader:
        # Train Discriminators
        fake_zebra = gen_z(horse)
        fake_horse = gen_h(zebra)
        D_loss = discriminator_loss(real, fake)
        
        # Train Generators
        cycle_horse = gen_h(fake_zebra)
        cycle_zebra = gen_z(fake_horse)
        G_loss = adversarial + cycle + identity
```

## Usage

1. Download the horse2zebra dataset
2. Update dataset paths in the code
3. Run all cells in sequence
4. Monitor training progress via tqdm bars
5. Check generated images in `/kaggle/working/images/`

## Notes

- Training uses GPU if available (`cuda` device)
- Instance normalization helps with style transfer
- Cycle consistency prevents mode collapse
- Identity loss can be enabled by setting `LAMDA_I > 0`

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- Original implementation: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## License

This implementation is for educational and research purposes.