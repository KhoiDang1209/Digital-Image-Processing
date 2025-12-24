import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        out, h = self.gru(x)      # h: (num_layers, batch, hidden_size)
        h_last = h[-1]            # (batch, hidden_size)
        logits = self.fc(h_last)  # (batch, num_classes)
        return logits
