import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# stacking dataset definition
class StackingDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values)
        if y is not None:
            y_val = y.values.flatten() if hasattr(y, 'values') else y
            self.y = torch.LongTensor(y_val)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# model definition
class StackingModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, output_dim=3):
        super(StackingModel, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.01
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x # cross entropy loss을 위한 logit
    
    def train_stacking_model(self, X_train, y_train, input_dim=6):
        self.dataset = StackingDataset(X_train, y_train)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StackingModel(input_dim=input_dim).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"Start Training Stacking Model (Input Dim: {input_dim})...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in self.dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(self.dataloader):.4f}")
