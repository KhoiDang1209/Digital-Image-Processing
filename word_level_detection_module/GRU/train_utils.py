import torch
from torch import nn, optim
from typing import Tuple


def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seq, labels in loader:
        # seq, labels từ DataLoader đã là Tensor trên CPU
        seq = seq.float().to(device)      # (B, T, F)
        labels = labels.long().to(device) # (B,)

        optimizer.zero_grad()
        outputs = model(seq)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seq.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def eval_one_epoch(model: nn.Module,
                   loader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.float().to(device)
            labels = labels.long().to(device)

            outputs = model(seq)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * seq.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc