import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

class LightGBM:
    def __init__(self, df, target_col):
        self.df = df.copy()
        self.y = self.df[target_col] - 1 # LightGBM은 타겟이 0부터 시작임
        
        cols_to_drop = [target_col]
        if 'building_id' in self.df.columns:
            cols_to_drop.append('building_id')    
        self.X = self.df.drop(columns=cols_to_drop, errors='ignore').copy()

        self.params = {
            "objective": "multiclass",
            "num_class": len(self.y.unique()),
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 128,
            "max_depth": -1, # 이걸 5~10 사이로 수정하면 더 안정적인 성능이 나온다고 함
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "min_gain_to_split": 0.0,
            "n_estimators": 2000,
            "random_state": 42,
        }

        self.models = []

    # LightGBM 전용 카테고리 타입 변환
    def encoding(self, categorical_cols):
        for col in categorical_cols:
            if col in self.X.columns:
                self.X[col] = self.X[col].astype("category")

    def kfold(self, n_splits):
        self.n_splits = n_splits
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.oof_preds = np.zeros((len(self.X), len(self.y.unique())))
        self.models = []

    def train(self):
        for fold, (train_idx, valid_idx) in tqdm(enumerate(self.kfold.split(self.X, self.y))):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_valid, y_valid = self.X.iloc[valid_idx], self.y.iloc[valid_idx]

            model = lgb.LGBMClassifier(**self.params)
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(100, verbose=True)]
            )

            self.oof_preds[valid_idx] = model.predict_proba(X_valid)
            self.models.append(model)
            print(f"[Fold {fold}] done.")

    def eval(self):
        self.y_pred = np.argmax(self.oof_preds, axis=1)
        
        # micro f1
        self.score_micro = f1_score(self.y, self.y_pred, average="micro")
        print(f"LightGBM Small Model CV Micro F1: {self.score_micro:.5f}")
        
        return self.score_micro

class LightGBMLimit(LightGBM):
    def __init__(self, df, target_col):
        super().__init__(df, target_col)
        self.params["max_depth"]=10
