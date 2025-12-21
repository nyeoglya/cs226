import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from model import StackingDataset
from sklearn.model_selection import StratifiedKFold

# 예측 결과 반환
def predict_prob(lgb_models, xgb_models, test_df, categorical_cols, train_cols):
    test_df = test_df.drop(columns=['building_id'], errors='ignore')
    lgb_test = test_df.copy()
    xgb_test = test_df.copy()
    
    n_classes = lgb_models[0].n_classes_ 

    # LGB
    for col in categorical_cols:
        lgb_test[col] = lgb_test[col].astype("category")
    
    # XGB
    xgb_test = pd.get_dummies(xgb_test, columns=categorical_cols, drop_first=False)
    xgb_test = xgb_test.reindex(columns=train_cols, fill_value=0)

    lgb_pred_avg = np.zeros((len(lgb_test), n_classes))
    for model in lgb_models:
        lgb_pred_avg += model.predict_proba(lgb_test)
    
    lgb_pred_avg /= len(lgb_models)
    
    lgb_cols = [f'lgb_class{c}' for c in range(n_classes)]
    lgb_result_df = pd.DataFrame(lgb_pred_avg, columns=lgb_cols)
    
    xgb_pred_avg = np.zeros((len(xgb_test), n_classes))
    
    for model in xgb_models:
        xgb_pred_avg += model.predict_proba(xgb_test)
        
    xgb_pred_avg /= len(xgb_models)
    
    xgb_cols = [f'xgb_class{c}' for c in range(n_classes)]
    xgb_result_df = pd.DataFrame(xgb_pred_avg, columns=xgb_cols)

    final_pred_df = pd.concat([lgb_result_df, xgb_result_df], axis=1)
    
    return final_pred_df

# lgb 예측
def predict_lgb(models, test_df, categorical_cols):
    X_test = test_df.copy()
    for col in categorical_cols:
        X_test[col] = X_test[col].astype("category")

    # f-kold 배열
    n_test = len(X_test)
    n_classes = models[0].n_classes_
    test_preds = np.zeros((n_test, n_classes))

    # 예측 수행
    for model in models:
        test_preds += model.predict_proba(X_test)

    return test_preds / len(models)

def predict_xgb(models, test_df, categorical_cols, train_cols):
    X_test = test_df.copy()
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

    # 훈련 데이터의 컬럼 순서 및 개수에 맞게 재정렬 (컬럼 일치)
    X_test = X_test.reindex(columns=train_cols, fill_value=0)
    
    # f-kold 배열
    n_test = len(X_test)
    n_classes = models[0].n_classes_ 
    test_preds = np.zeros((n_test, n_classes))

    # 예측 수행
    for model in models:
        test_preds += model.predict_proba(X_test) 

    return test_preds / len(models)

# meta data 생성
def get_test_meta_data(lgb_models, xgb_models, test_df, categorical_cols, train_cols):
    ids = test_df['building_id'] if 'building_id' in test_df.columns else None
    test_df = test_df.drop(columns=['building_id'], errors='ignore')
    
    lgb_test = test_df.copy()
    for col in categorical_cols:
        lgb_test[col] = lgb_test[col].astype("category")

    xgb_test = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)
    xgb_test = xgb_test.reindex(columns=train_cols, fill_value=0)
    
    n_classes = 3
    
    print("테스트 데이터 예측값 생성 중...")
    # LGBM 평균
    lgb_pred = np.zeros((len(test_df), n_classes))
    for model in lgb_models:
        lgb_pred += model.predict_proba(lgb_test)
    lgb_pred /= len(lgb_models)
    
    # XGB 평균
    xgb_pred = np.zeros((len(test_df), n_classes))
    for model in xgb_models:
        xgb_pred += model.predict_proba(xgb_test)
    xgb_pred /= len(xgb_models)
    
    # 컬럼 합치기
    lgb_cols = [f'lgb_class{c}' for c in range(n_classes)]
    xgb_cols = [f'xgb_class{c}' for c in range(n_classes)]
    
    meta_df = pd.concat([pd.DataFrame(lgb_pred, columns=lgb_cols), pd.DataFrame(xgb_pred, columns=xgb_cols)], axis=1)
    return meta_df, ids

# stacking 결과 반환
def predict_stacking(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    dataset = StackingDataset(X_test)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    
    with torch.no_grad():
        for X_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1) # softmax로 변환
            all_preds.append(probs.cpu().numpy())
            
    return np.vstack(all_preds)

# OOF 예측값 생성
def get_oof_predictions(lgb_models, xgb_models, X, y, categorical_cols, train_cols):
    X = X.drop(columns=['building_id'], errors='ignore')
    
    # LGBM
    lgb_X = X.copy()
    for col in categorical_cols:
        lgb_X[col] = lgb_X[col].astype("category")
        
    # XGB
    xgb_X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    xgb_X = xgb_X.reindex(columns=train_cols, fill_value=0)
    
    n_samples = len(X)
    n_classes = 3
    oof_lgb = np.zeros((n_samples, n_classes))
    oof_xgb = np.zeros((n_samples, n_classes))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_target = y.values.flatten() if hasattr(y, 'values') else y

    print("Generating OOF Predictions...")

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y_target)):
        X_val_lgb = lgb_X.iloc[val_idx]
        X_val_xgb = xgb_X.iloc[val_idx]
        model_lgb = lgb_models[i]
        model_xgb = xgb_models[i]
        oof_lgb[val_idx] = model_lgb.predict_proba(X_val_lgb)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val_xgb)
        
    lgb_cols = [f'lgb_class{c}' for c in range(n_classes)]
    xgb_cols = [f'xgb_class{c}' for c in range(n_classes)]
    
    df_lgb = pd.DataFrame(oof_lgb, columns=lgb_cols)
    df_xgb = pd.DataFrame(oof_xgb, columns=xgb_cols)
    
    return pd.concat([df_lgb, df_xgb], axis=1)
