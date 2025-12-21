import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

# 1. PyTorch 모델 정의 (내부용)
class _DREncoder(nn.Module):
    def __init__(self, latent_dim, sizes):
        super().__init__()
        # sizes = (n1, n2, n3)
        self.emb1 = nn.Embedding(sizes[0], 16)
        self.emb2 = nn.Embedding(sizes[1], 512)
        self.emb3 = nn.Embedding(sizes[2], 1024)
        self.compressor = nn.Linear(16 + 512 + 1024, latent_dim)

    def forward(self, x):
        e1 = self.emb1(x[:, 0])
        e2 = self.emb2(x[:, 1])
        e3 = self.emb3(x[:, 2])
        x_concat = torch.cat((e1, e2, e3), dim=1)
        x_latent = F.relu(x_concat)
        return self.compressor(x_latent)

class _DRDecoder(nn.Module):
    def __init__(self, latent_dim, sizes):
        super().__init__()
        self.head1 = nn.Linear(latent_dim, sizes[0])
        self.head2 = nn.Linear(latent_dim, sizes[1])
        self.head3 = nn.Linear(latent_dim, sizes[2])

    def forward(self, x):
        # x is latent vector
        return self.head1(x), self.head2(x), self.head3(x)

class _GeoAutoEncoderModule(nn.Module):
    def __init__(self, latent_dim, sizes):
        super().__init__()
        self.encoder = _DREncoder(latent_dim, sizes)
        self.decoder = _DRDecoder(latent_dim, sizes)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        return self.decoder(latent)


# 2. 사용자용 Wrapper 클래스 (Scikit-Learn 스타일)
class GeoAutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 latent_dim=16, 
                 epochs=10, 
                 batch_size=256, 
                 learning_rate=0.001,
                 device=None):
        """
        Args:
            latent_dim: 압축할 차원 크기
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            device: 'cuda' or 'cpu' (None일 경우 자동 감지)
        """
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.sizes = None # (max_id_1, max_id_2, max_id_3)
        self.cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    def _get_dataset(self, X):
        # DataFrame을 TensorDataset으로 변환
        data_tensor = torch.tensor(X[self.cols].values, dtype=torch.long)
        return TensorDataset(data_tensor)

    def fit(self, X, y=None):
        """
        DataFrame을 입력받아 모델을 학습합니다.
        """
        # 1. 데이터 크기 자동 감지 (ID + 1)
        self.sizes = (
            X[self.cols[0]].max() + 1,
            X[self.cols[1]].max() + 1,
            X[self.cols[2]].max() + 1
        )
        print(f"Detected sizes: L1={self.sizes[0]}, L2={self.sizes[1]}, L3={self.sizes[2]}")

        # 2. 데이터 로더 준비
        dataset = self._get_dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. 모델 초기화
        self.model = _GeoAutoEncoderModule(self.latent_dim, self.sizes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss: 각 레벨별 CrossEntropy의 합
        criterion = nn.CrossEntropyLoss()

        # 4. 학습 루프
        self.model.train()
        print(f"Training on {self.device} for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False):
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                recon1, recon2, recon3 = self.model(x)
                
                # 각 레벨별 재구성 오차 합산
                loss = (criterion(recon1, x[:, 0]) + 
                        criterion(recon2, x[:, 1]) + 
                        criterion(recon3, x[:, 2]))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")
            
        return self

    def transform(self, X):
        """
        학습된 모델을 사용하여 데이터를 임베딩 벡터로 변환합니다.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        
        self.model.eval()
        dataset = self._get_dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size*2, shuffle=False)
        
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Embeddings"):
                x = batch[0].to(self.device)
                
                # Encoder 통과 -> ReLU -> Latent Vector 추출
                encoded = self.model.encoder(x)
                latent = F.relu(encoded)
                
                embeddings.append(latent.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save(self, path):
        torch.save({
            'state_dict': self.model.state_dict(),
            'sizes': self.sizes,
            'latent_dim': self.latent_dim
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        # map_location은 이미 self.device로 설정되어 있습니다.
        map_device = self.device 
        
        # weights_only=False 추가: PyTorch가 numpy 객체를 로드할 수 있도록 허용
        checkpoint = torch.load(
            path, 
            map_location=map_device, 
            weights_only=False  # <--- 이 부분이 핵심
        )
        
        self.sizes = checkpoint['sizes']
        self.latent_dim = checkpoint['latent_dim']
        
        # 이후 모델 초기화 및 가중치 로드는 그대로 진행
        self.model = _GeoAutoEncoderModule(self.latent_dim, self.sizes).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        return self
