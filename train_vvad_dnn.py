#!/usr/bin/env python3
"""
Train a DNN for Visual Voice Activity Detection (V-VAD)
using facial landmark coordinates as input features.

Input:  Frontalized & normalized X, Y coordinates of 68 facial landmarks
        across a sliding window of consecutive video frames (default: 5 frames).
        Video frame rate: 30 FPS (confirmed from data timestamps).

Ground Truth: Audio-based VAD labels (0/1 per frame) from *_vad.csv files.
              VAD resolution in the CSV files matches the video frame rate (~30 FPS).

Preprocessing pipeline per frame:
    1. Face Frontalization (derotation) — builds a face-aligned coordinate system
       from stable reference landmarks (eye corners, nose bridge, chin) and applies
       the inverse rotation to all 68 landmarks.
    2. Centering — subtracts the midpoint between eye centers.
    3. Scale Normalization — divides by inter-ocular distance to remove the effect
       of the speaker's distance from the camera.

Usage:
    python train_vvad_dnn.py [--data_dir data_train] [--epochs 100] [--window_size 5]

Author: auto-generated training script
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from collections import OrderedDict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = "data_train"
DEFAULT_WINDOW_SIZE = 5           # Number of consecutive frames per sample
NUM_LANDMARKS = 68                # All 68 facial landmarks
VIDEO_FPS = 30.0                  # Confirmed from data timestamps (1/30 s intervals)

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_DIMS = [256, 128, 64]  # Three hidden layers + output
DEFAULT_DROPOUT = 0.3
DEFAULT_VAL_RATIO = 0.15          # Fraction of recordings for validation
DEFAULT_TEST_RATIO = 0.15         # Fraction of recordings for test
DEFAULT_MODEL_SAVE_PATH = "vvad_dnn_model.pt"
DEFAULT_HISTORY_PATH = "vvad_dnn_training_history.json"

# Stable reference landmark indices for building the face coordinate system
LEFT_EYE_OUTER = 36
LEFT_EYE_INNER = 39
RIGHT_EYE_INNER = 42
RIGHT_EYE_OUTER = 45
NOSE_BRIDGE = 27
CHIN = 8


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing: Face Frontalization & Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def frontalize_and_normalize_landmarks(pts_x, pts_y, pts_z):
    """
    Frontalize and normalize ALL 68 facial landmarks.

    The pipeline:
      1. Compute a face-aligned coordinate system from stable reference points
         (eye corners, nose bridge, chin).
      2. Apply the inverse of the face rotation matrix to all landmarks
         (= face frontalization / derotation).
      3. Center all landmarks around the midpoint between the two eye centers.
      4. Divide by the inter-ocular distance for scale invariance
         (removes dependence on the speaker's distance from the camera).

    Args:
        pts_x: np.ndarray of shape (68,) — X coordinates of all landmarks
        pts_y: np.ndarray of shape (68,) — Y coordinates of all landmarks
        pts_z: np.ndarray of shape (68,) — Z coordinates of all landmarks

    Returns:
        np.ndarray of shape (68, 2) with normalized (x, y) coordinates,
        or None if required reference points are invalid.
    """
    # Stack into (68, 3) array
    pts_3d = np.column_stack([pts_x, pts_y, pts_z])

    # Eye centers (stable across expressions)
    left_eye_center = (pts_3d[LEFT_EYE_OUTER] + pts_3d[LEFT_EYE_INNER]) / 2.0
    right_eye_center = (pts_3d[RIGHT_EYE_INNER] + pts_3d[RIGHT_EYE_OUTER]) / 2.0

    # X axis: left eye center → right eye center
    x_axis = right_eye_center - left_eye_center
    inter_ocular_dist = np.linalg.norm(x_axis)
    if inter_ocular_dist < 1e-8:
        return None
    x_axis = x_axis / inter_ocular_dist

    # Y approximate direction: nose bridge → chin (pointing downward)
    y_approx = pts_3d[CHIN] - pts_3d[NOSE_BRIDGE]

    # Z axis: perpendicular to face plane
    z_axis = np.cross(x_axis, y_approx)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return None
    z_axis = z_axis / z_norm

    # Y axis: orthogonal completion
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Rotation matrix: columns are the face coordinate axes in world coordinates
    # R_face transforms from face coords to world coords
    # R_inv = R_face^T transforms from world coords to face coords (= frontalization)
    R_inv = np.array([x_axis, y_axis, z_axis])  # (3, 3) — equivalent to R_face.T

    # Centering reference: midpoint between eye centers
    face_center = (left_eye_center + right_eye_center) / 2.0

    # Apply: center → derotate → scale-normalize → keep only X, Y
    centered = pts_3d - face_center                     # (68, 3)
    derotated = (R_inv @ centered.T).T                  # (68, 3)
    normalized_xy = derotated[:, :2] / inter_ocular_dist  # (68, 2)

    return normalized_xy


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def discover_recordings(data_dir):
    """Find all recording IDs available in data_dir.

    Looks for files matching the pattern: <rec_id>_vad.csv
    Returns a sorted list of recording IDs.
    """
    recordings = set()
    for fname in os.listdir(data_dir):
        if fname.endswith('_vad.csv'):
            rec_id = fname.replace('_vad.csv', '')
            recordings.add(rec_id)
    return sorted(recordings)


def load_single_recording(data_dir, rec_id, window_size=5):
    """
    Load and preprocess a single recording.

    Steps:
      1. Read mouth_position CSV (68 landmarks × 3 coords per frame) and VAD CSV.
      2. For each frame, frontalize & normalize the landmarks (face_idx=0 only).
      3. Build sliding windows of `window_size` consecutive valid frames.
      4. Align each window's center frame with the corresponding VAD label.

    Returns:
        features: np.ndarray of shape (N_samples, window_size * 68 * 2) or None
        labels:   np.ndarray of shape (N_samples,) with 0/1 VAD labels or None
    """
    mouth_path = os.path.join(data_dir, f"{rec_id}_mouth_position.csv")
    vad_path = os.path.join(data_dir, f"{rec_id}_vad.csv")

    if not os.path.exists(mouth_path) or not os.path.exists(vad_path):
        print(f"  [WARN] Missing files for {rec_id}, skipping.")
        return None, None

    df_mouth = pd.read_csv(mouth_path)
    df_vad = pd.read_csv(vad_path)

    # ── Filter to face_idx == 0 (primary / closest face) ──────────
    df_face0 = df_mouth[df_mouth['face_idx'] == 0].copy()

    # Get sorted unique timestamps
    timestamps = np.sort(df_face0['seconds'].unique())

    # Build VAD lookup arrays
    vad_times = df_vad['seconds'].values
    vad_vals = df_vad['vadDagcDecFinal'].values

    # ── Process each frame ────────────────────────────────────────
    # For efficiency, pivot the data: for each timestamp, get (68,) arrays of x, y, z
    # We group by timestamp and point_type, assuming point_type 0..67
    frame_features = []  # List of (68, 2) np arrays or None
    frame_labels = []    # List of 0/1

    for t in timestamps:
        frame_data = df_face0[df_face0['seconds'] == t]

        if len(frame_data) < NUM_LANDMARKS:
            frame_features.append(None)
        else:
            # Sort by point_type to ensure correct ordering
            frame_sorted = frame_data.sort_values('point_type')
            pts_x = frame_sorted['x'].values[:NUM_LANDMARKS].astype(np.float64)
            pts_y = frame_sorted['y'].values[:NUM_LANDMARKS].astype(np.float64)
            pts_z = frame_sorted['z'].values[:NUM_LANDMARKS].astype(np.float64)

            normalized = frontalize_and_normalize_landmarks(pts_x, pts_y, pts_z)
            frame_features.append(normalized)

        # Find closest VAD label by timestamp
        vidx = np.argmin(np.abs(vad_times - t))
        frame_labels.append(int(vad_vals[vidx]))

    # ── Build sliding windows with position + delta features ─────
    # For each window of W frames we compute:
    #   • Absolute positions for all W frames:  W × 68 × 2 = W*136 values
    #   • Frame-to-frame deltas (movement):    (W-1) × 68 × 2 = (W-1)*136 values
    # Total feature dim per sample = (2*W - 1) × 136
    half_w = window_size // 2
    features_list = []
    labels_list = []

    for i in range(half_w, len(frame_features) - half_w):
        # All frames in the window must have valid features
        window_frames = frame_features[i - half_w: i + half_w + 1]
        if any(f is None for f in window_frames):
            continue

        # Absolute positions: each frame (68, 2) → flatten (136,)
        abs_feats = np.concatenate([f.flatten() for f in window_frames])

        # Delta features: differences between consecutive frames
        deltas = []
        for j in range(1, len(window_frames)):
            delta = window_frames[j] - window_frames[j - 1]
            deltas.append(delta.flatten())
        delta_feats = np.concatenate(deltas)

        # Combine: [abs_positions | deltas]
        window_feat = np.concatenate([abs_feats, delta_feats])
        features_list.append(window_feat)
        labels_list.append(frame_labels[i])  # Label of the center frame

    if len(features_list) == 0:
        return None, None

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    return features, labels


def load_all_data(data_dir, window_size=5):
    """
    Load all recordings from data_dir.

    Returns a list of (features, labels, rec_id) tuples — one per valid recording.
    The recording-level structure is preserved for proper train/val/test splitting
    (splitting by recording avoids data leakage).
    """
    recordings = discover_recordings(data_dir)
    print(f"Found {len(recordings)} recordings in '{data_dir}'")

    all_data = []

    for rec_id in recordings:
        features, labels = load_single_recording(data_dir, rec_id, window_size)
        if features is not None:
            all_data.append((features, labels, rec_id))
            n_active = int(labels.sum())
            pct = 100 * labels.mean()
            print(f"  {rec_id}: {len(labels)} samples, "
                  f"active={n_active} ({pct:.1f}%)")
        else:
            print(f"  {rec_id}: SKIPPED (no valid data)")

    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class VVADDataset(Dataset):
    """Simple PyTorch dataset wrapping numpy feature/label arrays."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# DNN Model
# ═══════════════════════════════════════════════════════════════════════════════

class VVAD_DNN(nn.Module):
    """
    DNN for Visual Voice Activity Detection.

    Architecture:
        [Input] → (FC → BatchNorm → ReLU → Dropout) × N_hidden → FC → [logit]

    Default: 2 hidden layers (256, 128) + output = 3 layers total.

    Input:  Concatenated frontalized + normalized (x, y) landmark coordinates
            from a sliding window of consecutive frames.
            Dimension = window_size × 68 landmarks × 2 coordinates.
    Output: Single logit (use sigmoid for probability; BCEWithLogitsLoss for training).
    """

    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append((f'fc{i+1}', nn.Linear(prev_dim, h_dim)))
            layers.append((f'bn{i+1}', nn.BatchNorm1d(h_dim)))
            layers.append((f'relu{i+1}', nn.ReLU()))
            layers.append((f'drop{i+1}', nn.Dropout(dropout)))
            prev_dim = h_dim

        # Output layer (single logit)
        layers.append(('fc_out', nn.Linear(prev_dim, 1)))

        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    correct = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        n_samples += batch_size

    return total_loss / n_samples, correct / n_samples


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model. Returns (avg_loss, accuracy, preds, probs, labels)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels))


def print_metrics(labels, preds, title=""):
    """Print detailed binary classification metrics."""
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    print(f"  Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(labels, preds, zero_division=0):.4f}")

    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"              Silent  Speaking")
    print(f"  Actual Silent  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  Actual Speaking{cm[1,0]:5d}   {cm[1,1]:5d}")

    print(f"\n{classification_report(labels, preds, target_names=['Silent', 'Speaking'], zero_division=0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train a DNN for Visual Voice Activity Detection (V-VAD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Path to directory with training data CSVs')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help='Sliding window size (number of consecutive frames)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Initial learning rate (Adam)')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                        default=DEFAULT_HIDDEN_DIMS,
                        help='Hidden layer dimensions (e.g. 256 128)')
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT,
                        help='Dropout rate')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO,
                        help='Fraction of recordings for validation')
    parser.add_argument('--test_ratio', type=float, default=DEFAULT_TEST_RATIO,
                        help='Fraction of recordings for test')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_SAVE_PATH,
                        help='Path to save the best trained model')
    parser.add_argument('--history_path', type=str, default=DEFAULT_HISTORY_PATH,
                        help='Path to save training history JSON (for documentation)')
    parser.add_argument('--early_stop_patience', type=int, default=25,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--no_split', action='store_true',
                        help='Use ALL data for training (no val/test split). '
                             'Best model selected on training set metrics. '
                             'Early stopping is disabled.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Banner ────────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(f"  V-VAD DNN Training")
    print(f"{'═'*60}")
    print(f"  Device:        {device}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Window size:   {args.window_size} frames")
    print(f"  Video FPS:     {VIDEO_FPS:.0f} (confirmed from data timestamps)")
    print(f"  Hidden layers: {args.hidden_dims}")
    print(f"  Dropout:       {args.dropout}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  No split:      {args.no_split}")
    print(f"  Seed:          {args.seed}")
    print(f"{'═'*60}\n")

    # ── 1. Load Data ──────────────────────────────────────────────
    print("Step 1: Loading and preprocessing data...")
    print(f"  Preprocessing pipeline per frame:")
    print(f"    • Face frontalization (derotation via stable reference landmarks)")
    print(f"    • Centering (midpoint of eye centers as origin)")
    print(f"    • Scale normalization (÷ inter-ocular distance)")
    print()

    t0 = time.time()
    all_data = load_all_data(args.data_dir, args.window_size)
    t_load = time.time() - t0

    if len(all_data) == 0:
        print("ERROR: No valid recordings found! Check data_dir path.")
        sys.exit(1)

    total_samples = sum(len(d[1]) for d in all_data)
    total_active = sum(d[1].sum() for d in all_data)
    print(f"\n  Total: {len(all_data)} recordings, {total_samples} samples, "
          f"active={int(total_active)} ({100*total_active/total_samples:.1f}%)")
    print(f"  Data loading took {t_load:.1f}s")

    # ── 2. Split recordings into Train / Val / Test ───────────────
    #    (split by recording to avoid data leakage between frames
    #     of the same video appearing in different sets)

    no_split = args.no_split
    n_rec = len(all_data)
    indices = list(range(n_rec))

    def combine_splits(idxs):
        feats = np.concatenate([all_data[i][0] for i in idxs])
        labs = np.concatenate([all_data[i][1] for i in idxs])
        return feats, labs

    if no_split:
        # ── No-split mode: ALL data → training ──────────────────
        print(f"\nStep 2: Using ALL data for training (--no_split)")

        train_idx = indices
        val_idx = []
        test_idx = []

        X_train, y_train = combine_splits(train_idx)

        print(f"  Train: {len(train_idx):2d} recordings, {len(y_train):6d} samples "
              f"(active {100*y_train.mean():.1f}%)")
        print(f"  Val:   — (no validation set)")
        print(f"  Test:  — (no test set)")

        # Feature standardization
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0

        X_train = (X_train - feat_mean) / feat_std
        print(f"\n  Feature standardization applied (z-score, fit on all data)")

        train_rec_names = [all_data[i][2] for i in train_idx]
        val_rec_names = []
        test_rec_names = []
        print(f"\n  Train recordings: {train_rec_names}")

    else:
        # ── Standard mode: stratified Train / Val / Test split ──
        print(f"\nStep 2: Splitting data by recording...")

        # Stratify by activity bucket so train/val/test have similar class distributions
        activity_ratios = [float(d[1].mean()) for d in all_data]
        strat_labels = []
        for r in activity_ratios:
            if r < 0.25:
                strat_labels.append(0)
            elif r < 0.55:
                strat_labels.append(1)
            else:
                strat_labels.append(2)

        n_test = max(1, int(round(n_rec * args.test_ratio)))
        try:
            train_val_idx, test_idx = train_test_split(
                indices, test_size=n_test, random_state=args.seed,
                stratify=strat_labels)
        except ValueError:
            train_val_idx, test_idx = train_test_split(
                indices, test_size=n_test, random_state=args.seed)

        n_val = max(1, int(round(n_rec * args.val_ratio)))
        rel_val = min(0.5, n_val / len(train_val_idx))
        strat_remaining = [strat_labels[i] for i in train_val_idx]
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=rel_val, random_state=args.seed,
                stratify=strat_remaining)
        except ValueError:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=rel_val, random_state=args.seed)

        X_train, y_train = combine_splits(train_idx)
        X_val, y_val = combine_splits(val_idx)
        X_test, y_test = combine_splits(test_idx)

        print(f"  Train: {len(train_idx):2d} recordings, {len(y_train):6d} samples "
              f"(active {100*y_train.mean():.1f}%)")
        print(f"  Val:   {len(val_idx):2d} recordings, {len(y_val):6d} samples "
              f"(active {100*y_val.mean():.1f}%)")
        print(f"  Test:  {len(test_idx):2d} recordings, {len(y_test):6d} samples "
              f"(active {100*y_test.mean():.1f}%)")

        # Feature standardization (z-score) — fit on training set only
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0

        X_train = (X_train - feat_mean) / feat_std
        X_val = (X_val - feat_mean) / feat_std
        X_test = (X_test - feat_mean) / feat_std
        print(f"\n  Feature standardization applied (z-score, fit on train set)")

        train_rec_names = [all_data[i][2] for i in train_idx]
        val_rec_names = [all_data[i][2] for i in val_idx]
        test_rec_names = [all_data[i][2] for i in test_idx]
        print(f"\n  Train recordings: {train_rec_names}")
        print(f"  Val recordings:   {val_rec_names}")
        print(f"  Test recordings:  {test_rec_names}")

    # ── 3. Class weighting ────────────────────────────────────────
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.FloatTensor([n_neg / n_pos]).to(device)
    else:
        pos_weight = torch.FloatTensor([1.0]).to(device)
    print(f"\n  BCEWithLogitsLoss pos_weight: {pos_weight.item():.3f}")

    # ── 4. DataLoaders ────────────────────────────────────────────
    train_loader = DataLoader(VVADDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)

    if no_split:
        # Evaluation on training data (model in eval mode — no dropout/BN train)
        eval_loader = DataLoader(VVADDataset(X_train, y_train),
                                 batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = DataLoader(VVADDataset(X_val, y_val),
                                batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(VVADDataset(X_test, y_test),
                                 batch_size=args.batch_size, shuffle=False)

    # ── 5. Build Model ────────────────────────────────────────────
    # Feature dim = absolute positions + deltas:
    #   positions:  window_size × 68 × 2
    #   deltas:     (window_size - 1) × 68 × 2
    #   total:      (2 × window_size - 1) × 136
    n_pos_feats = args.window_size * NUM_LANDMARKS * 2
    n_delta_feats = (args.window_size - 1) * NUM_LANDMARKS * 2
    input_dim = n_pos_feats + n_delta_feats
    model = VVAD_DNN(input_dim, args.hidden_dims, args.dropout).to(device)

    print(f"\nStep 3: Model architecture")
    print(f"  Input dim: {input_dim}  "
          f"(positions: {n_pos_feats} + deltas: {n_delta_feats})")
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── 6. Training ───────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    if no_split:
        eval_label = "Train(eval)"
        print(f"\nStep 4: Training for {args.epochs} epochs (no early stopping, --no_split)...")
    else:
        eval_label = "Val"
        print(f"\nStep 4: Training for up to {args.epochs} epochs "
              f"(early stopping patience={args.early_stop_patience})...")

    print(f"{'─'*80}")
    print(f"{'Epoch':>6s}  {'Train Loss':>10s}  {'Train Acc':>9s}  "
          f"{eval_label+' Loss':>10s}  {eval_label+' Acc':>9s}  "
          f"{eval_label+' F1':>8s}  {'LR':>10s}")
    print(f"{'─'*80}")

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    t_train_start = time.time()

    # ── Epoch history for documentation / charts ──────────────────
    epoch_history = []

    # Choose the loader for evaluation (= train data in no-split mode)
    _eval_loader = eval_loader if no_split else val_loader

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate in eval mode (BN uses running stats, dropout off)
        v_loss, v_acc, v_preds, v_probs, v_labels = evaluate(
            model, _eval_loader, criterion, device)

        v_f1 = f1_score(v_labels, v_preds, zero_division=0)
        v_prec = precision_score(v_labels, v_preds, zero_division=0)
        v_rec = recall_score(v_labels, v_preds, zero_division=0)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_loss)

        # Record every epoch
        epoch_history.append({
            'epoch': epoch,
            'train_loss': round(t_loss, 6),
            'train_acc': round(t_acc, 6),
            'val_loss': round(v_loss, 6),
            'val_acc': round(v_acc, 6),
            'val_f1': round(v_f1, 6),
            'val_precision': round(v_prec, 6),
            'val_recall': round(v_rec, 6),
            'lr': current_lr,
        })

        # Print progress (every 10 epochs + first + last)
        print_interval = 10 if args.epochs > 100 else 5
        if epoch % print_interval == 0 or epoch == 1:
            print(f"{epoch:6d}  {t_loss:10.4f}  {t_acc:9.4f}  "
                  f"{v_loss:10.4f}  {v_acc:9.4f}  {v_f1:8.4f}  {current_lr:10.2e}")

        # Save best model (by eval F1, with loss as tiebreaker)
        if v_f1 > best_val_f1 or (v_f1 == best_val_f1 and v_loss < best_val_loss):
            best_val_f1 = v_f1
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_loss,
                'val_f1': v_f1,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dims': args.hidden_dims,
                    'dropout': args.dropout,
                    'window_size': args.window_size,
                    'num_landmarks': NUM_LANDMARKS,
                    'video_fps': VIDEO_FPS,
                },
                'standardization': {
                    'mean': feat_mean,
                    'std': feat_std,
                },
                'args': vars(args),
            }, args.model_path)
        else:
            patience_counter += 1

        # Early stopping only in split mode
        if not no_split and patience_counter >= args.early_stop_patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.early_stop_patience} epochs)")
            break

    last_epoch = epoch  # remember the final epoch number

    # ── Save last-epoch checkpoint ────────────────────────────────
    last_model_path = args.model_path.replace('.pt', '_last.pt')
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': v_loss,
        'val_f1': v_f1,
        'config': {
            'input_dim': input_dim,
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout,
            'window_size': args.window_size,
            'num_landmarks': NUM_LANDMARKS,
            'video_fps': VIDEO_FPS,
        },
        'standardization': {
            'mean': feat_mean,
            'std': feat_std,
        },
        'args': vars(args),
    }, last_model_path)

    t_train = time.time() - t_train_start
    metric_label = "training" if no_split else "validation"
    print(f"{'─'*80}")
    print(f"  Training complete in {t_train:.1f}s")
    print(f"  Best {metric_label} F1: {best_val_f1:.4f}  (loss: {best_val_loss:.4f}) at epoch {best_epoch}")
    print(f"  Last epoch: {last_epoch}  ({metric_label} F1: {v_f1:.4f}, loss: {v_loss:.4f})")
    print(f"  Best model saved to: {args.model_path}")
    print(f"  Last model saved to: {last_model_path}")

    # ── 7. Final Evaluation ──────────────────────────────────────
    if no_split:
        eval_set_label = "TRAINING SET (all data)"
        final_loader = eval_loader
        final_idx = train_idx
    else:
        eval_set_label = "TEST SET"
        final_loader = test_loader
        final_idx = test_idx

    print(f"\nStep 5: Evaluating best model on {eval_set_label}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded best model from epoch {checkpoint['epoch']}")

    _, test_acc, test_preds, test_probs, test_labels = evaluate(
        model, final_loader, criterion, device)

    print_metrics(test_labels, test_preds, title=f"{eval_set_label} — Overall Results")

    # ── Per-recording breakdown ───────────────────────────────────
    print(f"{'═'*60}")
    print(f"  Per-Recording Results ({eval_set_label})")
    print(f"{'═'*60}")
    offset = 0
    for idx in final_idx:
        _, labs, rec_id = all_data[idx]
        n = len(labs)
        rec_preds = test_preds[offset:offset + n]
        rec_labels = test_labels[offset:offset + n]
        r_acc = accuracy_score(rec_labels, rec_preds)
        r_f1 = f1_score(rec_labels, rec_preds, zero_division=0)
        r_prec = precision_score(rec_labels, rec_preds, zero_division=0)
        r_rec = recall_score(rec_labels, rec_preds, zero_division=0)
        print(f"  {rec_id:25s}  Acc={r_acc:.3f}  F1={r_f1:.3f}  "
              f"Prec={r_prec:.3f}  Rec={r_rec:.3f}  "
              f"[{int(rec_labels.sum())}/{n} active]")
        offset += n

    # ── 8. Save training history JSON (for documentation) ────────
    # Collect per-recording data distribution
    rec_info = []
    for feats, labs, rid in all_data:
        rec_info.append({
            'rec_id': rid,
            'n_samples': len(labs),
            'n_active': int(labs.sum()),
            'active_pct': round(100 * float(labs.mean()), 2),
        })

    # Collect per-recording results for the evaluated set
    per_rec_test = []
    offset = 0
    for idx in final_idx:
        _, labs, rec_id = all_data[idx]
        n = len(labs)
        rp = test_preds[offset:offset + n]
        rl = test_labels[offset:offset + n]
        per_rec_test.append({
            'rec_id': rec_id,
            'n_samples': n,
            'n_active': int(rl.sum()),
            'accuracy': round(float(accuracy_score(rl, rp)), 4),
            'f1': round(float(f1_score(rl, rp, zero_division=0)), 4),
            'precision': round(float(precision_score(rl, rp, zero_division=0)), 4),
            'recall': round(float(recall_score(rl, rp, zero_division=0)), 4),
        })
        offset += n

    cm = confusion_matrix(test_labels, test_preds)

    # Build data section — handle no-split gracefully
    data_section = {
        'total_recordings': len(all_data),
        'total_samples': total_samples,
        'total_active': int(total_active),
        'total_active_pct': round(100 * total_active / total_samples, 2),
        'no_split': no_split,
        'train_recordings': train_rec_names,
        'val_recordings': val_rec_names,
        'test_recordings': test_rec_names,
        'train_samples': len(y_train),
        'val_samples': len(y_val) if not no_split else 0,
        'test_samples': len(y_test) if not no_split else 0,
        'train_active_pct': round(100 * float(y_train.mean()), 2),
        'val_active_pct': round(100 * float(y_val.mean()), 2) if not no_split else 0.0,
        'test_active_pct': round(100 * float(y_test.mean()), 2) if not no_split else 0.0,
        'recordings': rec_info,
    }

    eval_results_label = "eval_results"  # generic label
    history_report = {
        'config': {
            'window_size': args.window_size,
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'weight_decay': 1e-4,
            'num_landmarks': NUM_LANDMARKS,
            'video_fps': VIDEO_FPS,
            'input_dim': input_dim,
            'n_position_features': n_pos_feats,
            'n_delta_features': n_delta_feats,
            'n_params': n_params,
            'pos_weight': round(pos_weight.item(), 4),
            'device': str(device),
            'seed': args.seed,
            'no_split': no_split,
            'early_stop_patience': args.early_stop_patience if not no_split else None,
        },
        'data': data_section,
        'epoch_history': epoch_history,
        'best_epoch': best_epoch,
        'last_epoch': last_epoch,
        'training_time_s': round(t_train, 1),
        'best_model_path': str(Path(args.model_path).name),
        'last_model_path': str(Path(last_model_path).name),
        'test_results': {
            'eval_set': 'train (all data)' if no_split else 'test',
            'accuracy': round(float(test_acc), 4),
            'precision': round(float(precision_score(test_labels, test_preds, zero_division=0)), 4),
            'recall': round(float(recall_score(test_labels, test_preds, zero_division=0)), 4),
            'f1': round(float(f1_score(test_labels, test_preds, zero_division=0)), 4),
            'confusion_matrix': {
                'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
            },
            'per_recording': per_rec_test,
        },
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.history_path, 'w') as f:
        json.dump(history_report, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Training history saved to: {os.path.abspath(args.history_path)}")
    print(f"  Best model saved to:  {os.path.abspath(args.model_path)}")
    print(f"  Last model saved to:  {os.path.abspath(last_model_path)}")
    print(f"  Done! ✓")


if __name__ == '__main__':
    main()
