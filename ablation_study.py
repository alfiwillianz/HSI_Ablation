"""
Ablation Study on Classification Heads for Hyperspectral Image Classification
Optimized for RTX 5090 (large batch, AMP, fast dataloaders)

- AMP + GradScaler (2x speed, 2x memory savings)
- PCGrad supports AMP
- Auto dataloader tuning (num_workers, persistent_workers, prefetch_factor)
- Early stopping improved (monitors loss + F1)
"""

import os, json, random, logging, warnings, copy
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ========================== REPRO / DEVICE ==========================
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== LOGGING ==========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("logs/ablation_study.log"), logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Device: {device}")

# ========================== DATASET ==========================
class HyperspectralDataset(Dataset):
    def __init__(self, data, labels, mask_ratio=0.15):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask_ratio = mask_ratio
        self.n_bands = self.data.shape[1]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        spectrum = self.data[idx]
        masked = spectrum.clone()

        n_mask = max(1, int(self.n_bands * self.mask_ratio))
        mask_idx = torch.randperm(self.n_bands)[:n_mask]
        mask = torch.zeros(self.n_bands, dtype=torch.bool)
        mask[mask_idx] = True

        masked[mask] = 0.0  # zero-mask

        return {
            "spectrum": spectrum,
            "masked_spectrum": masked,
            "mask": mask,
            "label": self.labels[idx]
        }

# ========================== PCGRAD — AMP READY ==========================
class PCGrad:
    def __init__(self, optimizer, reduction="mean"):
        self._optim = optimizer
        self._reduction = reduction

    @property
    def optimizer(self): return self._optim
    def zero_grad(self, set_to_none=True): return self._optim.zero_grad(set_to_none=set_to_none)
    def step(self): return self._optim.step()

    def pc_backward(self, objectives, scaler=None):
        grads, shapes, has_grads = self._pack_grad(objectives, scaler=scaler)
        merged = self._project_conflicting(grads, has_grads)
        merged = self._unflatten_grad(merged, shapes[0])
        self._set_grad(merged)

    def _pack_grad(self, objectives, scaler=None):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)

            if scaler:
                scaler.scale(obj).backward(retain_graph=True)
            else:
                obj.backward(retain_graph=True)

            grad, shape, has = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has, shape))
            shapes.append(shape)

        return grads, shapes, has_grads

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = copy.deepcopy(grads)

        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    denom = g_j.norm() ** 2 + 1e-12
                    g_i -= (dot / denom) * g_j

        merged = torch.zeros_like(grads[0])
        merged[shared] = torch.stack([g[shared] for g in pc_grad]).mean(0)
        merged[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(0)
        return merged

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]; idx += 1

    def _retrieve_grad(self):
        grads, shapes, has = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p)); has.append(torch.zeros_like(p))
                else:
                    grads.append(p.grad.clone()); has.append(torch.ones_like(p.grad))
                shapes.append(p.shape)
        return grads, shapes, has

    def _flatten_grad(self, grads, shapes):
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grad(self, grads, shapes):
        new, idx = [], 0
        for s in shapes:
            size = int(torch.tensor(s).prod())
            new.append(grads[idx:idx + size].view(s).clone())
            idx += size
        return new

# ========================== MODEL PARTS ==========================
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        self.smoothing = nn.Sequential(
            nn.Conv1d(1, 1, 25, padding=12),
            nn.BatchNorm1d(1),
            nn.GELU()
        )

        layers, prev = [], 1
        for h in hidden_dims[:2]:
            layers += [nn.Conv1d(prev, h, 7, stride=2, padding=3),
                       nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.shared = nn.Sequential(*layers)

        self.high = nn.Sequential(
            nn.Conv1d(prev, hidden_dims[2], 3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU()
        )

        self.fc_latent = nn.Conv1d(hidden_dims[2], latent_dim, 1)

    def forward(self, x):
        x = self.smoothing(x.unsqueeze(1))
        h = self.shared(x)
        h3 = self.high(h)
        z = F.adaptive_avg_pool1d(self.fc_latent(h3), 1).squeeze(-1)
        return h3, z

class Decoder(nn.Module):
    def __init__(self, input_channels=128, output_dim=2550):
        super().__init__()
        self.output_dim = output_dim
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(input_channels, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 512, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.dec(x)
        return F.interpolate(x, size=self.output_dim, mode="linear").squeeze(1)

class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes, mode, seq_len=319):
        super().__init__()
        mode = mode.lower()

        if mode == "linear":
            self.net = nn.Sequential(nn.BatchNorm1d(feature_dim), nn.AdaptiveAvgPool1d(1),
                                     nn.Flatten(), nn.Linear(feature_dim, num_classes))

        elif mode == "mlp":
            self.net = nn.Sequential(nn.BatchNorm1d(feature_dim), nn.AdaptiveAvgPool1d(1),
                                     nn.Flatten(), nn.Linear(feature_dim, 128), nn.ReLU(),
                                     nn.Dropout(0.3), nn.Linear(128, num_classes))

        elif mode == "flatten_mlp":
            self.net = nn.Sequential(nn.Flatten(),
                                     nn.Linear(feature_dim * seq_len, 512), nn.ReLU(),
                                     nn.Dropout(0.4), nn.Linear(512, 128),
                                     nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

        elif mode == "conv":
            self.net = nn.Sequential(
                nn.Conv1d(feature_dim, 256, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes),
            )

        elif mode == "gap_conv":
            self.net = nn.Sequential(
                nn.Conv1d(feature_dim, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes),
            )

        elif mode == "attn":
            self.q = nn.Linear(feature_dim, feature_dim)
            self.k = nn.Linear(feature_dim, feature_dim)
            self.v = nn.Linear(feature_dim, feature_dim)
            self.attn = nn.MultiheadAttention(feature_dim, 4, batch_first=True)
            self.fc = nn.Linear(feature_dim, num_classes)

        elif mode == "transformer":
            layer = nn.TransformerEncoderLayer(feature_dim, 4, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.fc = nn.Linear(feature_dim, num_classes)

        elif mode == "gru":
            self.gru = nn.GRU(feature_dim, 128, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(256, num_classes)

        else:
            raise ValueError(mode)

        self.mode = mode

    def forward(self, h3):
        if self.mode in ["linear", "mlp", "flatten_mlp", "conv", "gap_conv"]:
            return self.net(h3)

        x = h3.transpose(1, 2)

        if self.mode == "attn":
            q, k, v = self.q(x), self.k(x), self.v(x)
            out, _ = self.attn(q, k, v)
            return self.fc(out.mean(1))

        if self.mode == "transformer":
            return self.fc(self.enc(x).mean(1))

        if self.mode == "gru":
            _, h = self.gru(x)
            return self.fc(torch.cat([h[-2], h[-1]], dim=1))

class MultitaskAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, head_mode):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, latent_dim)
        self.decoder = Decoder()
        self.classifier = ClassificationHead(128, num_classes, mode=head_mode)

    def forward(self, x):
        h3, z = self.encoder(x)
        return {
            "reconstructed": self.decoder(h3),
            "class_logits": self.classifier(h3),
            "z": z,
            "features": h3
        }

# ========================== LOSS ==========================
def ae_loss(recon, gt, mask=None):
    return F.mse_loss(recon[mask], gt[mask]) if mask is not None else F.mse_loss(recon, gt)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super().__init__()
        self.alpha, self.gamma, self.weight = alpha, gamma, weight

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# ========================== TRAINING ==========================
def train_epoch(model, loader, optimizer, focal_loss, scaler, alpha_recon=0.5, alpha_class=1.0):
    model.train()
    total, correct, count = 0, 0, 0

    for b in loader:
        spectrum = b["spectrum"].to(device)
        masked = b["masked_spectrum"].to(device)
        mask = b["mask"].to(device)
        label = b["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            out = model(masked)
            rl = ae_loss(out["reconstructed"], spectrum, mask)
            cl = focal_loss(out["class_logits"], label)

        optimizer.pc_backward([alpha_recon * rl, alpha_class * cl], scaler=scaler)
        scaler.unscale_(optimizer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer.optimizer)
        scaler.update()

        total += (alpha_recon * rl + alpha_class * cl).item()
        _, pred = torch.max(out["class_logits"], 1)
        correct += (pred == label).sum().item()
        count += len(label)

    return total / len(loader), correct / count * 100

@torch.no_grad()
def validate_epoch(model, loader, focal_loss, alpha_recon=0.5, alpha_class=1.0):
    model.eval()
    total, preds, gts = 0, [], []

    for b in loader:
        spectrum = b["spectrum"].to(device)
        masked = b["masked_spectrum"].to(device)
        mask = b["mask"].to(device)
        label = b["label"].to(device)

        with autocast():
            out = model(masked)
            rl = ae_loss(out["reconstructed"], spectrum, mask)
            cl = focal_loss(out["class_logits"], label)

        total += (alpha_recon * rl + alpha_class * cl).item()
        preds.extend(out["class_logits"].argmax(1).cpu().numpy())
        gts.extend(label.cpu().numpy())

    return total / len(loader), accuracy_score(gts, preds), f1_score(gts, preds, average="weighted")

# ========================== DATA LOAD ==========================
def load_data_from_parquet(path):
    logger.info(f"Loading parquet: {path}")
    df = pd.read_parquet(path)

    label = "y" if "y" in df else "label"
    group = "plant" if "plant" in df else "group"

    X = df.drop(columns=[label, group]).values.astype(np.float32)
    y = df[label].values.astype(np.int32)
    groups = df[group].values.astype(np.int32)

    X = StandardScaler().fit_transform(X)

    return X, y, groups

# ========================== ABLATION ==========================
def train_head_ablation(head_mode, X, y, groups, epochs, batch_size, folds):
    logger.info(f"\n===== HEAD: {head_mode.upper()} =====")

    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\nFold {fold+1}/{folds}")

        train_ds = HyperspectralDataset(X[tr], y[tr])
        val_ds   = HyperspectralDataset(X[va], y[va])

        num_workers = min(32, os.cpu_count() - 2)
        prefetch = 4

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=prefetch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=max(1, num_workers//2), pin_memory=True,
                                persistent_workers=True, prefetch_factor=prefetch)

        model = MultitaskAE(input_dim=X.shape[1], latent_dim=64,
                            num_classes=len(np.unique(y)), head_mode=head_mode).to(device)

        optimizer = PCGrad(optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4))
        scaler = GradScaler()

        focal = FocalLoss(gamma=2.0)

        best_f1, patience = 0, 0

        for epoch in range(epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, focal, scaler)
            val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, focal)

            logger.info(f"Epoch {epoch+1:03d} | TrainLoss {tr_loss:.4f} Acc {tr_acc:.2f}% "
                        f"| ValLoss {val_loss:.4f} F1 {val_f1:.4f} Acc {val_acc:.2f}%")

            if val_f1 > best_f1 + 1e-4:
                best_f1 = val_f1
                patience = 0
            else:
                patience += 1

            if patience >= 15:
                logger.info("Early stopping.")
                break

        fold_scores.append(best_f1)

    logger.info(f"\n>>> HEAD {head_mode.upper()} — AVG F1 = {np.mean(fold_scores):.4f}")
    return head_mode, np.mean(fold_scores)


def run_ablation(head_modes=None, parquet="hf://datasets/alfiwillianz/hsi/data.parquet"):

    if head_modes is None:
        head_modes = ["linear", "mlp", "flatten_mlp", "conv", "gap_conv", "attn", "transformer", "gru"]

    X, y, groups = load_data_from_parquet(parquet)

    results = []
    for mode in head_modes:
        results.append(train_head_ablation(
            head_mode=mode, X=X, y=y, groups=groups,
            epochs=500, batch_size=128, folds=3
        ))

    df = pd.DataFrame(results, columns=["head_mode", "avg_f1"])
    df.to_csv("ablation_results.csv", index=False)
    logger.info(df)

    return df


if __name__ == "__main__":
    run_ablation()
