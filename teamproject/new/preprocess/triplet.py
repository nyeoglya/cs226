import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm

class GeoTripletDataset(Dataset):
    def __init__(self, df, l3_col='geo_level_3_id', l2_col='geo_level_2_id'):
        self.df = df.reset_index(drop=True)
        self.l3_col = l3_col
        self.l2_col = l2_col
        
        # 빠른 샘플링을 위해 L2 ID별로 L3 ID들을 미리 그룹화
        # groups = { l2_id: [l3_id_A, l3_id_B, ...], ... }
        self.groups = self.df.groupby(l2_col)[l3_col].unique().to_dict()
        self.l2_ids = list(self.groups.keys())
        
        # 데이터프레임의 각 행을 Anchor 후보로 사용
        self.anchors = self.df[[l3_col, l2_col]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Anchor 선정
        anchor_l3, anchor_l2 = self.anchors[idx]
        
        # 2. Positive 선정 (같은 L2 그룹 내에서 랜덤 선택)
        # 자기 자신이 뽑힐 수도 있지만, 데이터가 많으면 확률이 낮으므로 단순화
        pos_candidates = self.groups[anchor_l2]
        if len(pos_candidates) > 1:
            pos_l3 = np.random.choice(pos_candidates)
        else:
            pos_l3 = anchor_l3 # 그룹에 하나밖에 없으면 자기 자신 선택
            
        # 3. Negative 선정 (다른 L2 그룹에서 랜덤 선택)
        # 현재 Anchor의 L2가 아닌 다른 L2를 무작위로 하나 고름
        while True:
            neg_l2 = np.random.choice(self.l2_ids)
            if neg_l2 != anchor_l2:
                break
        
        neg_l3 = np.random.choice(self.groups[neg_l2])
        
        return {
            "anchor": torch.tensor(anchor_l3, dtype=torch.long),
            "positive": torch.tensor(pos_l3, dtype=torch.long),
            "negative": torch.tensor(neg_l3, dtype=torch.long)
        }

class GeoMetricEncoder(nn.Module):
    def __init__(self, num_l3, embedding_dim=64, hidden_dim=128):
        super().__init__()
        # 1. 기초 임베딩 (Look-up Table)
        self.embedding = nn.Embedding(num_l3, hidden_dim)
        
        # 2. 프로젝션 헤드 (Projection Head)
        # 임베딩을 더 의미 있는 공간으로 변환
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.projection(x)
        
        # 3. L2 Normalization (Metric Learning의 핵심!)
        # 벡터의 길이를 1로 맞춰서, 벡터 간의 '각도(방향)' 차이만 거리로 측정되게 함
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def train_metric(df, num_l3, epochs=5, batch_size=256, embedding_dim=32, device='cuda'):
    # 데이터셋 및 로더 준비
    dataset = GeoTripletDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 준비
    model = GeoMetricEncoder(num_l3=num_l3, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Triplet Loss 정의
    # margin=1.0: Negative가 Positive보다 최소 1.0만큼 더 멀리 떨어지게 강제함
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            anc = batch['anchor'].to(device)
            pos = batch['positive'].to(device)
            neg = batch['negative'].to(device)
            
            optimizer.zero_grad()
            
            # 3개의 임베딩 추출
            emb_anc = model(anc)
            emb_pos = model(pos)
            emb_neg = model(neg)
            
            # Loss 계산: d(a, p) < d(a, n) - margin
            loss = criterion(emb_anc, emb_pos, emb_neg)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
        
    return model

def metric_embed(model, l3_ids, device):
    model.eval()
    l3_tensor = torch.tensor(l3_ids, dtype=torch.long).to(device)
    
    with torch.no_grad():
        embeddings = model(l3_tensor).cpu().numpy()
        
    return embeddings
