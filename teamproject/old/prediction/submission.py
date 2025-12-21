import numpy as np
import pandas as pd

from prediction import predict_xgb
from prediction import predict_lgb

def create_xgb_submission(models, test_df, test_building_ids, categorical_cols, train_cols):
    print("테스트 데이터 예측 확률 계산 중")

    avg_test_probabilities = predict_xgb(
        models=models,
        test_df=test_df.drop(columns=['building_id']),
        categorical_cols=categorical_cols,
        train_cols=train_cols,
    )

    predicted_classes_0_indexed = np.argmax(avg_test_probabilities, axis=1)
    predicted_damage_grade = predicted_classes_0_indexed + 1 # +1
    submission_df = pd.DataFrame({
        'building_id': test_building_ids.values,
        'damage_grade': predicted_damage_grade
    })
    
    print(f"xgb submission: {len(submission_df)}개 예측 결과 생성됨")
    
    return submission_df

def create_lgb_submission(models, test_df, test_building_ids, categorical_cols):
    print("테스트 데이터 예측 확률 계산 중")

    avg_test_probabilities = predict_lgb(
        models=models,
        test_df=test_df.drop(columns=['building_id']),
        categorical_cols=categorical_cols,
    )

    # 최종 클래스 예측
    predicted_classes_0_indexed = np.argmax(avg_test_probabilities, axis=1)
    predicted_damage_grade = predicted_classes_0_indexed + 1 # +1

    # 제출용 dataframe 생성
    submission_df = pd.DataFrame({
        'building_id': test_building_ids.values,
        'damage_grade': predicted_damage_grade
    })
    
    print(f"lgb submission: {len(submission_df)}개 예측 결과 생성됨")
    
    return submission_df

def create_ensemble_submission(lgb_models, xgb_models, test_df, test_building_ids, categorical_cols, train_cols, weights=(0.5, 0.5)):
    # XGBoost
    print("xgb 테스트 데이터 예측 확률 계산 중")
    avg_test_probabilities_xgb = predict_xgb(
        models=xgb_models,
        test_df=test_df.drop(columns=['building_id'], errors='ignore'),
        categorical_cols=categorical_cols,
        train_cols=train_cols
    )

    # LightGBM
    print("lgb 테스트 데이터 예측 확률 계산 중")
    avg_test_probabilities_lgb = predict_lgb( 
        models=lgb_models,
        test_df=test_df.drop(columns=['building_id'], errors='ignore'),
        categorical_cols=categorical_cols
    )

    print("모델 결과 가중 합산 중")
    final_probabilities = (avg_test_probabilities_lgb * weights[0]) + (avg_test_probabilities_xgb * weights[1])
    predicted_classes_0_indexed = np.argmax(final_probabilities, axis=1)
    predicted_damage_grade = predicted_classes_0_indexed + 1 # +1

    # 제출용 dataframe 생성
    submission_df = pd.DataFrame({
        'building_id': test_building_ids.values,
        'damage_grade': predicted_damage_grade
    })
    
    print(f"ensemble submission: {len(submission_df)}개 예측 결과 생성됨")
    
    return submission_df