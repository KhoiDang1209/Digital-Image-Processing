import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm


class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        sequence = np.load(row['filepath'])
        sequence = torch.FloatTensor(sequence)
        
        label = int(row['label'])
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label


class SignLanguageDatasetCached(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        print(f"Preloading {len(self.data)} sequences into memory...")
        self.sequences = []
        self.labels = []
        
        for idx in tqdm(range(len(self.data)), desc="Loading data"):
            row = self.data.iloc[idx]
            sequence = np.load(row['filepath'])
            self.sequences.append(torch.FloatTensor(sequence))
            self.labels.append(int(row['label']))
        
        print(f"✓ Loaded {len(self.sequences)} sequences into memory")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label
