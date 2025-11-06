# HSI Ablation Study - Classification Head Analysis

This project performs a comprehensive ablation study on different classification head architectures for hyperspectral image (HSI) classification using a multitask autoencoder.

## Overview

The ablation study evaluates the following classification head architectures:
- **Linear**: Simple linear layer with adaptive pooling
- **MLP**: Multi-layer perceptron with batch normalization
- **Flatten MLP**: Flattens features before MLP layers
- **Conv**: Convolutional feature extraction before classification
- **GAP Conv**: Global average pooling with convolution
- **Attention**: Multi-head self-attention mechanism
- **Transformer**: Transformer encoder layers
- **GRU**: Bidirectional GRU for sequence modeling

## Features

✨ **Key Capabilities:**
- Loads data from Hugging Face parquet dataset (`alfiwillianz/hsi`)
- Stratified group k-fold cross-validation
- Focal loss for handling class imbalance
- PCGrad optimizer for multitask learning
- Comprehensive metrics logging to CSV
- Automatic model checkpointing
- Detailed logging to console and file

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login to Hugging Face (for parquet dataset access):
```bash
huggingface-cli login
```

## Usage

### Run Full Ablation Study

```bash
python ablation_study.py
```

### Customize Parameters

Edit the `if __name__ == "__main__"` section in `ablation_study.py`:

```python
results_df, all_results = run_ablation_study(
    parquet_path="hf://datasets/alfiwillianz/hsi/data.parquet",
    head_modes=['linear', 'mlp', 'conv', 'attn', 'transformer'],  # Test specific heads
    epochs=100,
    batch_size=32,
    n_splits=5,
    latent_dim=128
)
```

## Output Files

The script generates the following outputs in the `results/` directory:

- **ablation_results.csv**: Summary metrics for each head (accuracy, F1 score)
- **ablation_results_detailed.json**: Detailed metrics including per-fold results
- **logs/ablation_study.log**: Complete training log
- **{head_mode}_fold{n}_best.pth**: Checkpoint files for each fold and head

## Results Format

### ablation_results.csv
```
head_mode,avg_accuracy,std_accuracy,avg_f1,std_f1
transformer,0.8523,0.0234,0.8512,0.0245
attn,0.8412,0.0312,0.8398,0.0328
mlp,0.8301,0.0289,0.8287,0.0301
...
```

### ablation_results_detailed.json
```json
[
  {
    "head_mode": "linear",
    "avg_accuracy": 0.8234,
    "std_accuracy": 0.0156,
    "avg_f1": 0.8221,
    "std_f1": 0.0167,
    "fold_metrics": [
      {
        "fold": 1,
        "accuracy": 0.8312,
        "f1_score": 0.8301
      },
      ...
    ]
  },
  ...
]
```

## Project Structure

```
HSI_Ablation/
├── ablation_study.py          # Main ablation study script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── results/                   # Output directory
│   ├── ablation_results.csv
│   ├── ablation_results_detailed.json
│   └── {head_mode}_fold{n}_best.pth
└── logs/
    └── ablation_study.log
```

## Model Architecture

### Encoder
- Input smoothing layer (learnable low-pass filter)
- Progressive downsampling with Conv1d layers
- 3 levels of feature extraction: [512, 256, 128] channels
- Output: Feature map (h3) + Latent vector (z)

### Decoder
- ConvTranspose1d layers for upsampling
- Reconstructs original spectra from h3
- Tanh activation for normalized reflectance [-1, 1]

### Classification Heads
Each head architecture processes the shared feature map h3 differently:
- **Simple heads** (linear, mlp): Direct processing with adaptive pooling
- **Feature extraction heads** (conv, gap_conv): Additional convolutional layers
- **Sequence heads** (attn, transformer, gru): Treat features as sequences

## Training Details

- **Loss**: Focal Loss (γ=2) for classification + MSE for reconstruction
- **Optimizer**: AdamW with PCGrad for multitask gradient projection
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Early Stopping**: Patience=10 epochs based on validation F1 score
- **Data Augmentation**: 15% spectral band masking

## Performance Notes

- GRU head typically requires more memory due to bidirectional processing
- Attention-based heads (attn, transformer) generally show best performance
- Linear head provides baseline for comparison
- Training time varies by head complexity: Linear < MLP < Transformer/GRU

## Contributing

To extend this ablation study:

1. Add new head architectures to `ClassificationHead.__init__()`
2. Update the `head_modes` list in `run_ablation_study()`
3. Run the study and compare results

## License

This project is part of the HSI analysis pipeline.

## Citation

If you use this ablation study, please cite:
```
@software{hsi_ablation_2025,
  title={HSI Classification Head Ablation Study},
  author={Your Name},
  year={2025}
}
```

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` (try 32 or 16)
- Reduce `n_splits` (try 2 or 3)
- Use smaller `latent_dim` (try 32)

### Parquet Loading Issues
```bash
# Ensure you're logged in to Hugging Face
huggingface-cli login

# Or specify local parquet path
results_df, all_results = run_ablation_study(
    parquet_path="/path/to/local/data.parquet"
)
```

### GPU Not Found
The script will automatically fall back to CPU. For GPU support:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Support

For issues or questions, check the logs in `logs/ablation_study.log` for detailed error messages.
