import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

class XGBoost:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.y = self.df[target_col].astype(int) - 1
        
        cols_to_drop = [target_col]
        if 'building_id' in self.df.columns:
            cols_to_drop.append('building_id')
        self.X = self.df.drop(columns=cols_to_drop, errors='ignore').copy()

        self.params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "eval_metric": 'mlogloss',
            "early_stopping_rounds": 500,
            "random_state": 43,
            # "use_label_encoder": False,
        }

        self.models = []
        self.n_classes = len(self.y.unique())
        self.oof_preds = np.zeros((len(self.X), self.n_classes))
        self.kfold_splitter = None
        self.n_splits = 0

    def encoding(self, categorical_cols: list):
        existing_cols = [col for col in categorical_cols if col in self.X.columns]
        if not existing_cols:
            return
        self.X = pd.get_dummies(self.X, columns=existing_cols, drop_first=False)
    
    def kfold(self, n_splits: int):
        self.n_splits = n_splits
        self.kfold_splitter = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=42
        )
    
    def train(self):
        if self.kfold_splitter is None:
            raise ValueError("self.kfold_splitter is None")
        
        for fold, (train_idx, valid_idx) in enumerate(self.kfold_splitter.split(self.X, self.y)):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_valid, y_valid = self.X.iloc[valid_idx], self.y.iloc[valid_idx]

            model = xgb.XGBClassifier(**self.params) 

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
            )
            
            self.oof_preds[valid_idx] = model.predict_proba(X_valid)
            self.models.append(model)
            
            best_score = model.best_score
            print(f"[Fold {fold+1}/{self.n_splits}] Best Score: {best_score:.4f}")

    def eval(self):
        self.y_pred = np.argmax(self.oof_preds, axis=1)
        self.score_micro = f1_score(self.y, self.y_pred, average="micro")
        
        print(f"XGBoost K-Fold CV Micro F1 Score: {self.score_micro:.5f}")
        return self.score_micro
