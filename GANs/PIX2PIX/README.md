# Pix2Pix: Image-to-Image Translation with Conditional GANs

A PyTorch Lightning implementation of Pix2Pix for paired image-to-image translation, converting satellite imagery to map views (and vice versa).

## Overview

Pix2Pix is a conditional GAN (cGAN) that learns a mapping from input images to output images using paired training examples. This implementation uses the Maps dataset to translate between satellite images and street maps.

## Architecture

### Generator (U-Net)
- **Encoder-Decoder with Skip Connections**
- Encoder:
  - Initial: Conv2d (3→64 channels) + LeakyReLU
  - 6 downsampling blocks: [64, 128, 256, 512, 512, 512]
  - Each block: Conv2d + BatchNorm + LeakyReLU + Dropout
- Bottleneck:
  - Conv2d (512→512) + ReLU
- Decoder:
  - 6 upsampling blocks (ConvTranspose2d + BatchNorm + ReLU)
  - First 3 blocks include 50% dropout
  - Skip connections concatenate encoder features
  - Final layer: ConvTranspose2d + Tanh
- **Output**: 256×256×3 RGB image

### Discriminator (PatchGAN)
- **70×70 Patch Discriminator**
- Takes both input and target images (6 channels concatenated)
- Architecture:
  - Initial: Conv2d (6→64) + LeakyReLU
  - 3 convolutional blocks: [128, 256, 512]
  - Final: Conv2d (512→1)
- **Output**: 30×30 patch predictions

## Key Features

- **Conditional Generation**: Generator is conditioned on input images
- **PatchGAN Discriminator**: Classifies 70×70 patches as real/fake
- **U-Net Generator**: Preserves spatial information via skip connections
- **L1 Loss**: Ensures output images are close to ground truth
- **PyTorch Lightning**: Simplified training with automatic optimization

## Requirements

```python
torch >= 1.12.0
torchvision
pytorch-lightning >= 2.0.0
albumentations >= 1.3.0
numpy
pandas
PIL
torchinfo
```

Install dependencies:
```bash
pip install torch torchvision pytorch-lightning albumentations torchinfo pillow
```

## Dataset Structure

```
maps/
├── train/
│   ├── 1.jpg  # Concatenated [input|target] images (600×600)
│   ├── 2.jpg
│   └── ...
└── val/
    ├── 1.jpg
    └── ...
```

**Note**: Each image is 1200×600 pixels (input: 600×600, target: 600×600 concatenated horizontally)

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 256×256 | Resized from 600×600 |
| Batch Size | 16 | Training batch size |
| Epochs | 100-500 | Recommended training duration |
| Learning Rate | 2e-4 | Same for both G and D |
| λ_L1 | 100 | L1 loss weight |
| Optimizer | Adam | β1=0.5, β2=0.999 |
| Precision | 16-mixed | Mixed precision training |
| Grad Accumulation | 4 batches | Generator only |

## Loss Functions

### Discriminator Loss
```
L_D = 0.5 × [BCE(D(x, y), 1) + BCE(D(x, G(x)), 0)]
```
- Distinguishes real pairs (x, y) from fake pairs (x, G(x))

### Generator Loss
```
L_G = BCE(D(x, G(x)), 1) + λ_L1 × L1(G(x), y)
```
- **Adversarial Loss**: Fools discriminator
- **L1 Loss**: Pixel-wise similarity to ground truth (weighted by λ_L1=100)

## Data Augmentation

```python
# Applied to both input and target
both_transform = A.Compose([
    A.Resize(256, 256)
])

# Applied separately with normalization
transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ToTensorV2()
])
```

## Training Process

1. **Discriminator Training:**
   - Generate fake image: `y_fake = G(x)`
   - Real prediction: `D(x, y_real)`
   - Fake prediction: `D(x, y_fake.detach())`
   - Update discriminator with combined loss

2. **Generator Training (with gradient accumulation):**
   - Generate fake image: `y_fake = G(x)`
   - Get discriminator feedback: `D(x, y_fake)`
   - Compute adversarial + L1 loss
   - Accumulate gradients over 4 batches
   - Update generator weights

3. **Image Saving:**
   - Saves input, generated, and target images every 10 batches
   - Images denormalized from [-1, 1] to [0, 1]

## Usage

### Basic Training

```python
# Initialize models
disc = Discriminator(in_channels=3)
gen = Generator(in_channels=3)

# Create Pix2Pix Lightning module
model = Pix2Pix(
    gen=gen,
    disc=disc,
    lr_gen=2e-4,
    lr_disc=2e-4,
    lambda_l1=100,
    image_save_dir="/path/to/save/images"
)

# Train
trainer = pl.Trainer(
    devices=1,
    max_epochs=100,
    precision='16-mixed'
)
trainer.fit(model, train_loader)
```

### Loading Saved Checkpoints

```python
# Load model from checkpoint
model = Pix2Pix.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# Generate predictions
with torch.no_grad():
    output = model(input_image)
```

## Output Structure

```
project/
├── image/
│   ├── input_0.png       # Input satellite images
│   ├── generated_0.png   # Generated map images
│   ├── target_0.png      # Ground truth maps
│   └── ...
└── lightning_logs/       # PyTorch Lightning logs
    └── version_0/
        ├── checkpoints/  # Model checkpoints
        └── hparams.yaml
```

## Key Implementation Details

### Manual Optimization
- Uses `automatic_optimization = False` for custom training logic
- Allows gradient accumulation for the generator only
- Discriminator updated every batch
- Generator updated every 4 batches

### Skip Connections
- U-Net architecture preserves spatial details
- Encoder features concatenated to decoder layers
- Helps maintain structural consistency between input and output

### PatchGAN Design
- 70×70 receptive field per output patch
- More efficient than full-image discrimination
- Encourages high-frequency detail

### Mixed Precision Training
- Uses `precision='16-mixed'` for faster training
- Reduces memory usage
- Maintains model quality

## Training Features

- **Automatic Checkpointing**: PyTorch Lightning saves best models
- **Progress Monitoring**: Loss logged to progress bar and TensorBoard
- **Image Saving**: Visual progress tracking every 10 batches
- **Gradient Accumulation**: Simulates larger batch sizes with limited GPU memory
- **Mixed Precision**: Faster training with automatic loss scaling

## Training Metrics

Monitor these losses during training:
- **loss_disc**: Discriminator loss (should stabilize around 0.5)
- **gen_loss**: Combined generator loss (should decrease over time)
- **L1 component**: Dominates early training (due to λ=100)
- **Adversarial component**: Becomes more important later

## Common Training Patterns

### Healthy Training:
```
Epoch 1  | loss_disc: 0.65 | gen_loss: 85.2
Epoch 10 | loss_disc: 0.52 | gen_loss: 45.8
Epoch 50 | loss_disc: 0.48 | gen_loss: 25.3
```

### Issues to Watch:
- **Discriminator too strong**: loss_disc < 0.1 → reduce discriminator LR
- **Generator collapse**: gen_loss plateaus → increase L1 lambda
- **Poor quality**: Increase training epochs or adjust L1 weight

## Customization

### Change Dataset
```python
# Update image splitting logic in MapDataset
input_image = image[:, :new_width, :]
target_image = image[:, new_width:, :]
```

### Adjust Architecture
```python
# Modify generator depth
gen = Generator(in_channels=3, features=[64, 128, 256, 512])
```

### Change L1 Weight
```python
# Higher values (200+) for more faithful reconstruction
# Lower values (50-) for more creative outputs
model = Pix2Pix(gen, disc, lr_gen=2e-4, lr_disc=2e-4, lambda_l1=200)
```

## Evaluation

Generate predictions on validation set:
```python
model.eval()
with torch.no_grad():
    for input_img, target_img in val_loader:
        fake_img = model(input_img)
        # Save or visualize results
```

## Tips for Better Results

1. **Train Longer**: 100-500 epochs for high-quality results
2. **Balanced Training**: Monitor discriminator/generator loss ratio
3. **L1 Weight**: Start with λ=100, adjust based on output quality
4. **Gradient Accumulation**: Use if GPU memory is limited
5. **Data Augmentation**: Add random flips/rotations for better generalization
6. **Learning Rate Scheduling**: Add LR decay for stable convergence

## Applications

This Pix2Pix implementation can be adapted for:
- Satellite → Map translation
- Sketch → Photo rendering
- Day → Night conversion
- Segmentation mask → Photo
- Black & white → Color images
- Edges → Photos

## References

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004): Image-to-Image Translation with Conditional Adversarial Networks
- [Original Implementation](https://github.com/phillipi/pix2pix): PyTorch implementation by authors
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)

## Performance Notes

- **Training Time**: ~1 hour per 100 epochs on T4 GPU
- **Memory Usage**: ~8GB VRAM with batch_size=16, mixed precision
- **Quality Improvements**: Visible after 20-30 epochs
- **Convergence**: Typically 100-200 epochs for good results

## License

This implementation is for educational and research purposes.

## Troubleshooting

**Out of Memory?**
- Reduce batch size to 8 or 4
- Increase gradient accumulation steps
- Use smaller image resolution (128×128)

**Poor Image Quality?**
- Train for more epochs (200+)
- Increase L1 lambda to 150-200
- Add more data augmentation

**Mode Collapse?**
- Verify discriminator isn't too strong
- Check that both losses are updating
- Try adjusting learning rates

**Training Too Slow?**
- Enable mixed precision (`precision='16-mixed'`)
- Use multiple GPUs with `devices='auto'`
- Reduce image save frequency