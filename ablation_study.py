"""
Ablation Study on Classification Heads for Hyperspectral Image Classification
Using Multitask Autoencoder with different head architectures

This script:
- Loads data from Hugging Face parquet dataset
- Trains models with different classification head configurations
- Logs metrics to CSV for comparison
- Runs cross-validation for each head type
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
from datetime import datetime
from typing import List, Dict, Tuple
import logging
import copy
import random

warnings.filterwarnings("ignore", category=UserWarning)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configure logging
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/ablation_study.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========== DATASET CLASS ==========
class HyperspectralDataset(Dataset):
    """Custom Dataset for Hyperspectral Data"""

    def __init__(self, data: np.ndarray, labels: np.ndarray, mask_ratio: float = 0.15):
        """
        Args:
            data: Hyperspectral data (N_samples, N_bands)
            labels: Class labels (N_samples,)
            mask_ratio: Ratio of spectral bands to mask for reconstruction task
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.mask_ratio = mask_ratio
        self.n_bands = data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrum = self.data[idx]
        label = self.labels[idx]

        # Create masked spectrum for reconstruction task
        masked_spectrum = spectrum.clone()
        n_mask = int(self.n_bands * self.mask_ratio)

        # Random masking
        mask_indices = torch.randperm(self.n_bands)[:n_mask]
        mask = torch.zeros(self.n_bands, dtype=torch.bool)
        mask[mask_indices] = True

        masked_spectrum[mask] = 0.0

        return {
            'spectrum': spectrum,
            'masked_spectrum': masked_spectrum,
            'mask': mask,
            'label': label
        }


# ========== PCGRAD OPTIMIZER ==========
class PCGrad():
    """Projected Conflict Gradient for multitask learning"""
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


# ========== MODEL COMPONENTS ==========
class SharedEncoder(nn.Module):
    """Shared encoder with input-stage smoothing for noisy hyperspectral data."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()

        self.smoothing = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(1),
            nn.GELU()
        )

        prev_channels = 1
        conv_layers = []
        for hidden_dim in hidden_dims[:2]:
            conv_layers.extend([
                nn.Conv1d(prev_channels, hidden_dim, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_channels = hidden_dim
        self.shared_layers = nn.Sequential(*conv_layers)

        self.third_conv = nn.Sequential(
            nn.Conv1d(prev_channels, hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_latent = nn.Conv1d(hidden_dims[2], latent_dim, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.smoothing(x)
        h = self.shared_layers(x)
        h3 = self.third_conv(h)
        z = self.fc_latent(h3)
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
        return h3, z


class Decoder(nn.Module):
    """ConvTranspose1d decoder reconstructing spectra."""

    def __init__(self, input_channels: int = 128, output_dim: int = 2550):
        super().__init__()
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(input_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.ConvTranspose1d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.ConvTranspose1d(512, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.output_activation = nn.Tanh()

    def forward(self, h3):
        x = self.decoder(h3)
        x = F.interpolate(x, size=self.output_dim, mode="linear", align_corners=False)
        return self.output_activation(x.squeeze(1))


class ClassificationHead(nn.Module):
    """
    Configurable classification head for ablation study.
    Modes: ['linear', 'mlp', 'flatten_mlp', 'conv', 'gap_conv', 'attn', 'transformer', 'gru']
    """
    def __init__(self, feature_dim: int, num_classes: int, seq_len: int = 319, mode: str = "linear"):
        super().__init__()
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

        if self.mode == "linear":
            self.head = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, num_classes)
            )

        elif self.mode == "mlp":
            self.head = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )

        elif self.mode == "flatten_mlp":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim * seq_len, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        elif self.mode == "conv":
            self.head = nn.Sequential(
                nn.Conv1d(feature_dim, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )

        elif self.mode == "gap_conv":
            self.head = nn.Sequential(
                nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )

        elif self.mode == "attn":
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
            self.fc = nn.Linear(feature_dim, num_classes)

        elif self.mode == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.fc = nn.Linear(feature_dim, num_classes)

        elif self.mode == "gru":
            self.gru = nn.GRU(feature_dim, 128, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(256, num_classes)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, h3):
        if self.mode in ["linear", "mlp", "flatten_mlp", "conv", "gap_conv"]:
            return self.head(h3)

        elif self.mode == "attn":
            x = h3.transpose(1, 2)
            q, k, v = self.query_proj(x), self.key_proj(x), self.value_proj(x)
            attn_out, _ = self.attn(q, k, v)
            pooled = attn_out.mean(1)
            return self.fc(pooled)

        elif self.mode == "transformer":
            x = h3.transpose(1, 2)
            enc = self.encoder(x)
            pooled = enc.mean(1)
            return self.fc(pooled)

        elif self.mode == "gru":
            x = h3.transpose(1, 2)
            _, h = self.gru(x)
            h = torch.cat([h[-2], h[-1]], dim=1)
            return self.fc(h)


class MultitaskAE(nn.Module):
    """
    Dual-head multitask autoencoder with configurable classification head.
    """

    def __init__(self, input_dim: int, latent_dim: int, num_classes: int,
                 head_mode: str = "linear", hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(input_channels=hidden_dims[-1], output_dim=input_dim)
        self.classifier = ClassificationHead(
            feature_dim=hidden_dims[-1], 
            num_classes=num_classes,
            mode=head_mode
        )

    def forward(self, x):
        h3, z = self.encoder(x)
        reconstructed = self.decoder(h3)
        class_logits = self.classifier(h3)
        return {
            "reconstructed": reconstructed,
            "class_logits": class_logits,
            "z": z,
            "features": h3
        }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ========== DATA LOADING ==========
def load_data_from_parquet(parquet_path: str = "hf://datasets/alfiwillianz/hsi/data.parquet"):
    """
    Load data from Hugging Face parquet dataset.
    
    Returns:
        X: Spectral data (N_samples, N_bands)
        y: Class labels (N_samples,)
        groups: Group identifiers for stratification (N_samples,)
    """
    logger.info(f"Loading data from {parquet_path}...")
    
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Assume 'label' or 'y' column for class labels, 'group' or 'plant' for groups
        label_col = 'y' if 'y' in df.columns else 'label'
        group_col = 'plant' if 'plant' in df.columns else 'group'
        
        # Get spectral columns (all columns except label and group)
        spectral_cols = [col for col in df.columns if col not in [label_col, group_col]]
        
        X = df[spectral_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int32)
        groups = df[group_col].values.astype(np.int32)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        groups = groups[valid_mask]
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        logger.info(f"Data loaded successfully!")
        logger.info(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        
        return X, y, groups, scaler
        
    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")
        raise


def ae_loss(recon_x, x, mask=None):
    """Autoencoder reconstruction loss with optional masking"""
    if mask is not None:
        recon_loss = F.mse_loss(recon_x[mask], x[mask], reduction='mean')
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    return recon_loss


# ========== TRAINING FUNCTIONS ==========
def train_epoch(model, train_loader, optimizer, focal_loss, alpha_recon=1.0, alpha_class=1.0):
    """Single training epoch"""
    model.train()
    train_loss = train_recon_loss = train_class_loss = 0
    train_correct = train_total = 0

    for batch in train_loader:
        spectrum = batch['spectrum'].to(device)
        masked_spectrum = batch['masked_spectrum'].to(device)
        mask = batch['mask'].to(device)
        labels_batch = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(masked_spectrum)

        recon_loss = ae_loss(outputs['reconstructed'], spectrum, mask)
        class_loss = focal_loss(outputs['class_logits'], labels_batch)
        total_loss = alpha_recon * recon_loss + alpha_class * class_loss

        optimizer.pc_backward([recon_loss, class_loss])
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += total_loss.item()
        train_recon_loss += recon_loss.item()
        train_class_loss += class_loss.item()

        _, predicted = torch.max(outputs['class_logits'].data, 1)
        train_total += labels_batch.size(0)
        train_correct += (predicted == labels_batch).sum().item()

    return {
        'loss': train_loss / len(train_loader),
        'recon_loss': train_recon_loss / len(train_loader),
        'class_loss': train_class_loss / len(train_loader),
        'accuracy': 100 * train_correct / train_total
    }


def validate_epoch(model, val_loader, focal_loss, alpha_recon=1.0, alpha_class=1.0):
    """Single validation epoch"""
    model.eval()
    val_loss = val_recon_loss = val_class_loss = 0
    val_correct = val_total = 0
    val_predictions = val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            spectrum = batch['spectrum'].to(device)
            masked_spectrum = batch['masked_spectrum'].to(device)
            mask = batch['mask'].to(device)
            labels_batch = batch['label'].to(device)

            outputs = model(masked_spectrum)
            recon_loss = ae_loss(outputs['reconstructed'], spectrum, mask)
            class_loss = focal_loss(outputs['class_logits'], labels_batch)
            total_loss = alpha_recon * recon_loss + alpha_class * class_loss

            val_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_class_loss += class_loss.item()

            _, predicted = torch.max(outputs['class_logits'].data, 1)
            val_total += labels_batch.size(0)
            val_correct += (predicted == labels_batch).sum().item()
            
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels_batch.cpu().numpy())

    return {
        'loss': val_loss / len(val_loader),
        'recon_loss': val_recon_loss / len(val_loader),
        'class_loss': val_class_loss / len(val_loader),
        'accuracy': 100 * val_correct / val_total,
        'f1_score': f1_score(val_targets, val_predictions, average='weighted')
    }


def train_head_ablation(head_mode: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       epochs: int = 50, batch_size: int = 64, n_splits: int = 3,
                       latent_dim: int = 64, output_dir: str = './results'):
    """
    Train model with specific classification head and return metrics.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Training with HEAD: {head_mode.upper()}")
    logger.info(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = []
    test_predictions_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\nFold {fold + 1}/{n_splits} - Head: {head_mode}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create dataloaders
        train_dataset = HyperspectralDataset(X_train, y_train, mask_ratio=0.15)
        val_dataset = HyperspectralDataset(X_val, y_val, mask_ratio=0.15)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup model
        model = MultitaskAE(input_dim, latent_dim, num_classes, head_mode=head_mode).to(device)
        optimizer = PCGrad(optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Setup loss
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        focal_loss = FocalLoss(alpha=1, gamma=2, weight=class_weights_tensor, reduction='mean')
        
        # Training loop
        best_val_loss = float('inf')
        best_val_f1 = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, focal_loss)
            val_metrics = validate_epoch(model, val_loader, focal_loss)
            
            scheduler.step(val_metrics['loss'])
            
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                model_path = f'{output_dir}/{head_mode}_fold{fold+1}_best.pth'
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Evaluate on validation set
        model.load_state_dict(torch.load(f'{output_dir}/{head_mode}_fold{fold+1}_best.pth'))
        model.eval()
        
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                masked_spectrum = batch['masked_spectrum'].to(device)
                labels_batch = batch['label'].to(device)
                
                outputs = model(masked_spectrum)
                _, predicted = torch.max(outputs['class_logits'].data, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())
        
        fold_accuracy = accuracy_score(val_targets, val_predictions)
        fold_f1 = f1_score(val_targets, val_predictions, average='weighted')
        
        logger.info(f"  Fold {fold+1} - Accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f}")
        
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'f1_score': fold_f1
        })
        
        test_predictions_per_fold.append(val_predictions)
    
    # Calculate average metrics
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in fold_metrics])
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    std_f1 = np.std([m['f1_score'] for m in fold_metrics])
    
    logger.info(f"\nHead: {head_mode.upper()}")
    logger.info(f"  Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"  Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    
    return {
        'head_mode': head_mode,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'fold_metrics': fold_metrics
    }


# ========== MAIN ABLATION STUDY ==========
def run_ablation_study(parquet_path: str = "hf://datasets/alfiwillianz/hsi/data.parquet",
                      head_modes: List[str] = None,
                      epochs: int = 50,
                      batch_size: int = 64,
                      n_splits: int = 3,
                      latent_dim: int = 64):
    """
    Run comprehensive ablation study on different classification heads.
    """
    
    if head_modes is None:
        head_modes = ['linear', 'mlp', 'flatten_mlp', 'conv', 'gap_conv', 'attn', 'transformer', 'gru']
    
    logger.info("="*70)
    logger.info("STARTING ABLATION STUDY ON CLASSIFICATION HEADS")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Heads to test: {head_modes}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, N-splits: {n_splits}")
    logger.info(f"Latent dim: {latent_dim}, Device: {device}")
    
    # Load data
    X, y, groups, scaler = load_data_from_parquet(parquet_path)
    
    # Create results directory
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run ablation for each head
    all_results = []
    for head_mode in head_modes:
        try:
            result = train_head_ablation(
                head_mode=head_mode,
                X=X, y=y, groups=groups,
                epochs=epochs,
                batch_size=batch_size,
                n_splits=n_splits,
                latent_dim=latent_dim,
                output_dir=results_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error training head {head_mode}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'head_mode': r['head_mode'],
            'avg_accuracy': r['avg_accuracy'],
            'std_accuracy': r['std_accuracy'],
            'avg_f1': r['avg_f1'],
            'std_f1': r['std_f1']
        }
        for r in all_results
    ])
    
    # Sort by F1 score
    results_df = results_df.sort_values('avg_f1', ascending=False)
    
    # Save results
    results_csv_path = f'{results_dir}/ablation_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"\n✅ Results saved to {results_csv_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*70)
    logger.info(results_df.to_string(index=False))
    logger.info("="*70)
    
    # Save detailed results
    detailed_results_path = f'{results_dir}/ablation_results_detailed.json'
    with open(detailed_results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"✅ Detailed results saved to {detailed_results_path}")
    
    return results_df, all_results


if __name__ == "__main__":
    # Run ablation study
    results_df, all_results = run_ablation_study(
        parquet_path="hf://datasets/alfiwillianz/hsi/data.parquet",
        epochs=50,
        batch_size=64,
        n_splits=3,
        latent_dim=64
    )
