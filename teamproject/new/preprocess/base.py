import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# torch dataset
class GeoBaseDataset(Dataset):
    def __init__(self, df, label_col=None):
        self.x1 = torch.tensor(df["geo_level_1_id_enc"].values, dtype=torch.long)
        self.x2 = torch.tensor(df["geo_level_2_id_enc"].values, dtype=torch.long)
        self.x3 = torch.tensor(df["geo_level_3_id_enc"].values, dtype=torch.long)
        
        if label_col is not None:
            self.y = torch.tensor(df[label_col].values - 1, dtype=torch.long) # -1
        else:
            self.y = None # 레이블 없으면 test

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x1[idx], self.x2[idx], self.x3[idx], self.y[idx]
        else:
            return self.x1[idx], self.x2[idx], self.x3[idx]
    
    def get_embedding(model, x1, x2, x3):
        e1 = model.embed_1(x1)
        e2 = model.embed_2(x2)
        e3 = model.embed_3(x3)
        return torch.cat([e1, e2, e3], dim=1)

# 기본 인코더 구조
class GeoBaseEncoder(nn.Module):
    def __init__(self, n1, n2, n3, d1, d2, d3, n_class):
        super().__init__()
        self.embed_1 = nn.Embedding(n1, d1)
        self.embed_2 = nn.Embedding(n2, d2)
        self.embed_3 = nn.Embedding(n3, d3)
        
        total_dim = d1 + d2 + d3
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_class)
        )
    
    def forward(self, x1, x2, x3):
        e1 = self.embed_1(x1)
        e2 = self.embed_2(x2)
        e3 = self.embed_3(x3)
        cat = torch.cat([e1, e2, e3], dim=1)
        return self.classifier(cat)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total, correct = 0, 0, 0
    for x1, x2, x3, y in loader:
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x1, x2, x3)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = y.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for x1, x2, x3, y in loader:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            out = model(x1, x2, x3)
            val_loss += criterion(out, y).item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

def epochs(
    model,
    num_epochs,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:02d}] "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        ldim_avg_val_loss = val_loss/len(val_loader)

        # save model
        if ldim_avg_val_loss < ldim_best_val_loss:
            ldim_best_val_loss = ldim_avg_val_loss
            ldim_best_model_state = model.state_dict()
            print(f"  -> Best Model Saved! (Loss: {ldim_best_val_loss:.4f})")

    # load best model
    print("Loading Best Model...")
    model.load_state_dict(ldim_best_model_state)

# base embedding 출력
def base_embed(df, model, device, batch_size=1024):
    dataset = GeoBaseDataset(df, label_col=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for x1, x2, x3 in tqdm(loader, desc="Extracting"):
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
            emb_vec = model.get_embedding(model, x1, x2, x3)
            embeddings_list.append(emb_vec.cpu().numpy())
    final_embeddings = np.concatenate(embeddings_list, axis=0)
    
    embed_cols = [f"geo_emb_{i}" for i in range(final_embeddings.shape[1])]
    embed_df = pd.DataFrame(final_embeddings, columns=embed_cols)
    
    df_out = pd.concat([df.reset_index(drop=True), embed_df], axis=1)
    df_out = df_out.drop(columns=[
        "geo_level_1_id",
        "geo_level_2_id",
        "geo_level_3_id",
        "geo_level_1_id_enc",
        "geo_level_2_id_enc",
        "geo_level_3_id_enc",
    ])
    
    return df_out
