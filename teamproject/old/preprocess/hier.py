import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# torch dataset
class GeoHierDataset(Dataset):
    def __init__(self, df, label_col):
        # 반드시 geo_level_1_id, geo_level_2_id, geo_level_3_id, label_col 있어야 됨
        self.df = df.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        geo1 = int(row["geo_level_1_id"])
        geo2 = int(row["geo_level_2_id"])
        geo3 = int(row["geo_level_3_id"])
        y = int(row[self.label_col])
        return {
            "geo1": torch.tensor(geo1, dtype=torch.long),
            "geo2": torch.tensor(geo2, dtype=torch.long),
            "geo3": torch.tensor(geo3, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
        }

# 개선된 인코더 구조
class GeoAttentionEncoder(nn.Module):
    def __init__(self, n1, n2, n3, d=32, geo_dim=32, n_heads=1):
        super().__init__()
        # 라벨 내용 임베딩
        self.embed_1 = nn.Embedding(n1, d)
        self.embed_2 = nn.Embedding(n2, d)
        self.embed_3 = nn.Embedding(n3, d)

        # 계층 레벨 임베딩
        self.level_embed = nn.Embedding(3, d)

        # 상위 -> 하위 계층
        self.f12 = nn.Linear(d, d) # e1 -> e2 보정
        self.f23 = nn.Linear(2 * d, d) # [e1, e2'] -> e3 보정

        # self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            batch_first=True,
        )

        # 토큰 3개의 출력을 pooled 해서 최종 geo 벡터로 압축
        self.geo_mlp = nn.Sequential(
            nn.Linear(d, geo_dim),
            nn.ReLU(),
            nn.LayerNorm(geo_dim),
        )

    def forward(self, geo1, geo2, geo3):
        B = geo1.size(0)

        e1 = self.embed_1(geo1)
        e2 = self.embed_2(geo2)
        e3 = self.embed_3(geo3)

        e2 = e2 + self.f12(e1)
        e3 = e3 + self.f23(torch.cat([e1, e2], dim=1))

        level_ids = torch.arange(3, device=geo1.device) # (3,)
        level_emb = self.level_embed(level_ids) # (3, d)
        level_emb = level_emb.unsqueeze(0).expand(B, -1, -1) # (B, 3, d)

        seq = torch.stack([e1, e2, e3], dim=1)  # (B, 3, d)
        seq = seq + level_emb # (B, 3, d)
        out, _ = self.attn(seq, seq, seq) # (B, 3, d)
        pooled = out.mean(dim=1) # (B, d)
        geo_vec = self.geo_mlp(pooled) # (B, geo_dim)
        return geo_vec

# wrapper class
class GeoEncoderWithHead(nn.Module):
    def __init__(self, n1, n2, n3, n_classes,
                 d=32, geo_dim=32, n_heads=1):
        super().__init__()
        self.encoder = GeoAttentionEncoder(
            n1=n1, n2=n2, n3=n3,
            d=d,
            geo_dim=geo_dim,
            n_heads=n_heads,
        )
        self.head = nn.Linear(geo_dim, n_classes)

    def forward(self, geo1, geo2, geo3):
        geo_vec = self.encoder(geo1, geo2, geo3) # (B, geo_dim)
        logits = self.head(geo_vec) # (B, n_classes)
        return logits

    # embedding 결과 반환
    def get_embedding(self, geo1, geo2, geo3):
        with torch.no_grad():
            return self.encoder(geo1, geo2, geo3)

# train
def hier_train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total, correct = 0, 0, 0
    for batch in loader:
        geo1 = batch["geo1"].to(device)
        geo2 = batch["geo2"].to(device)
        geo3 = batch["geo3"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(geo1, geo2, geo3) # (B, n_classes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# evaluation
def hier_eval(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            geo1 = batch["geo1"].to(device)
            geo2 = batch["geo2"].to(device)
            geo3 = batch["geo3"].to(device)
            labels = batch["label"].to(device)

            logits = model(geo1, geo2, geo3)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# hier 임베딩 출력
'''
def hier_embed(df, model, device, batch_size=1024):
    dataset = GeoHierDataset(df, label_col=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for x1, x2, x3 in tqdm(loader, desc="Extracting"):
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
            emb_vec = model.get_embedding(x1, x2, x3)
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
'''
 