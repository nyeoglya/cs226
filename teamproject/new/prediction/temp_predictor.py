import numpy as np
import pandas as pd

class Predictor:
    def __init__(self, lgb_models=None, xgb_models=None):
        self.lgb_models = lgb_models if lgb_models else []
        self.xgb_models = xgb_models if xgb_models else []
        self.n_classes = 3

    def predict_proba_lgb(self, test_df, categorical_cols):
        X_test = test_df.copy()
        for col in categorical_cols:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype("category")
        
        n_test = len(X_test)
        preds = np.zeros((n_test, self.n_classes))
        
        if not self.lgb_models:
            return preds
            
        for model in self.lgb_models:
            preds += model.predict_proba(X_test)
            
        return preds / len(self.lgb_models)

    def predict_proba_xgb(self, test_df, categorical_cols, train_cols):
        X_test = test_df.copy()
        
        # One-hot encoding for XGB
        existing_cat_cols = [c for c in categorical_cols if c in X_test.columns]
        X_test = pd.get_dummies(X_test, columns=existing_cat_cols, drop_first=False)
        
        # Align columns
        X_test = X_test.reindex(columns=train_cols, fill_value=0)
        
        n_test = len(X_test)
        preds = np.zeros((n_test, self.n_classes))
        
        if not self.xgb_models:
            return preds
            
        for model in self.xgb_models:
            preds += model.predict_proba(X_test)
            
        return preds / len(self.xgb_models)

    def predict_ensemble(self, test_df, categorical_cols, train_cols, weights=(0.5, 0.5)):
        # Remove building_id if present
        ids = test_df['building_id'] if 'building_id' in test_df.columns else None
        X_test = test_df.drop(columns=['building_id'], errors='ignore')
        
        lgb_probs = self.predict_proba_lgb(X_test, categorical_cols)
        xgb_probs = self.predict_proba_xgb(X_test, categorical_cols, train_cols)
        
        final_probs = (lgb_probs * weights[0]) + (xgb_probs * weights[1])
        return final_probs, ids

    def create_submission(self, test_df, categorical_cols, train_cols, weights=(0.5, 0.5), output_path="submission.csv"):
        print("Calculating predictions...")
        final_probs, ids = self.predict_ensemble(test_df, categorical_cols, train_cols, weights)
        
        predicted_classes = np.argmax(final_probs, axis=1) + 1 # 1-based index
        
        if ids is None:
            raise ValueError("building_id column missing in test_df")
            
        submission = pd.DataFrame({
            'building_id': ids,
            'damage_grade': predicted_classes
        })
        
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path} ({len(submission)} rows)")
        return submission
