import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total, correct = 0, 0, 0
    
    # Check if loader returns dict or tuple
    first_batch = next(iter(loader))
    is_dict = isinstance(first_batch, dict)
    
    for batch in loader:
        if is_dict:
            # Hierarchical dataset returns dict
            geo1 = batch["geo1"].to(device)
            geo2 = batch["geo2"].to(device)
            geo3 = batch["geo3"].to(device)
            y = batch["label"].to(device)
            out = model(geo1, geo2, geo3)
        else:
            # Base dataset returns tuple
            x1, x2, x3, y = batch
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            out = model(x1, x2, x3)
        
        optimizer.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct = 0, 0, 0
    
    first_batch = next(iter(loader))
    is_dict = isinstance(first_batch, dict)
    
    with torch.no_grad():
        for batch in loader:
            if is_dict:
                geo1 = batch["geo1"].to(device)
                geo2 = batch["geo2"].to(device)
                geo3 = batch["geo3"].to(device)
                y = batch["label"].to(device)
                out = model(geo1, geo2, geo3)
            else:
                x1, x2, x3, y = batch
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                out = model(x1, x2, x3)
                
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total
