import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# ===== PATH =====
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_NPY_ROOT = PROJECT_ROOT / "data_npy"


def build_label_map_from_data(root_dir: Path):
    classes = []
    for d in sorted(os.listdir(root_dir)):
        class_dir = root_dir / d
        if class_dir.is_dir() and any(f.endswith(".npy") for f in os.listdir(class_dir)):
            classes.append(d)
    return {name: idx for idx, name in enumerate(classes)}


def resize_seq(seq, target_len):
    T, D = seq.shape
    if T == target_len:
        return seq
    idx = np.linspace(0, T - 1, target_len)
    return np.stack([
        np.interp(idx, np.arange(T), seq[:, d])
        for d in range(D)
    ], axis=1)


def temporal_augment(seq):
    T, D = seq.shape

    # ---- A) random temporal crop ----
    crop_ratio = np.random.uniform(0.7, 1.0)
    new_T = max(4, int(T * crop_ratio))
    start = np.random.randint(0, T - new_T + 1)
    seq = seq[start:start + new_T]

    # ---- B) speed jitter ----
    speed = np.random.uniform(0.85, 1.2)
    target_T = max(4, int(len(seq) * speed))
    idx = np.linspace(0, len(seq) - 1, target_T)
    seq = np.stack([
        np.interp(idx, np.arange(len(seq)), seq[:, d])
        for d in range(D)
    ], axis=1)

    # ---- C) frame drop ----
    keep = np.random.rand(len(seq)) > 0.05
    if keep.sum() >= 4:
        seq = seq[keep]

    # ---- D) noise ----
    seq += np.random.normal(0, 0.01, seq.shape)

    return seq


class WLASLKeypointDataset(Dataset):
    def __init__(self, root_dir=DATA_NPY_ROOT, train=True, target_len=32):
        self.root_dir = Path(root_dir)
        self.train = train
        self.target_len = target_len

        self.label_map = build_label_map_from_data(self.root_dir)
        self.num_classes = len(self.label_map)

        self.samples = []
        for label_name, idx in self.label_map.items():
            class_dir = self.root_dir / label_name
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((class_dir / fname, idx))

        if len(self.samples) == 0:
            raise RuntimeError("Dataset rỗng – kiểm tra data_npy")

        print(f"[Dataset] Loaded {len(self.samples)} samples")
        print(f"[Dataset] Num classes = {self.num_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)  # (T, 63)

        if self.train:
            seq = temporal_augment(seq)

        seq = resize_seq(seq, self.target_len)

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)
