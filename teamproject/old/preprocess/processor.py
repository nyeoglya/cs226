import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, train_values_path, train_labels_path, test_values_path):
        self.train_values_path = train_values_path
        self.train_labels_path = train_labels_path
        self.test_values_path = test_values_path
        
        self.train_values = None
        self.train_labels = None
        self.test_values = None
        self.full_train = None
        
        self.categorical_cols = [
            "land_surface_condition", "foundation_type", "roof_type",
            "ground_floor_type", "other_floor_type", "position",
            "plan_configuration", "legal_ownership_status"
        ]
        self.numerical_cols = [
            "count_floors_pre_eq", "age", "area_percentage", "height_percentage",
            "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
            "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
            "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
            "has_superstructure_timber", "has_superstructure_bamboo",
            "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
            "has_superstructure_other", "count_families",
            "has_secondary_use", "has_secondary_use_agriculture",
            "has_secondary_use_hotel", "has_secondary_use_rental",
            "has_secondary_use_institution", "has_secondary_use_school",
            "has_secondary_use_industry", "has_secondary_use_health_post",
            "has_secondary_use_gov_office", "has_secondary_use_use_police",
            "has_secondary_use_other"
        ]
        self.geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
        
        self.preprocessor = None

    def load_data(self):
        self.train_values = pd.read_csv(self.train_values_path)
        self.train_labels = pd.read_csv(self.train_labels_path)
        self.test_values = pd.read_csv(self.test_values_path)
        
        self.full_train = self.train_values.merge(self.train_labels, on="building_id")
        
    def get_preprocessor(self):
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='passthrough'  # Keep other columns like geo_ids for now
        )
        return self.preprocessor

    def preprocess(self):
        if self.full_train is None:
            self.load_data()
            
        X = self.full_train.drop(columns=['damage_grade', 'building_id'])
        y = self.full_train['damage_grade']
        
        X_test = self.test_values.drop(columns=['building_id'])
        
        # Fit and transform
        if self.preprocessor is None:
            self.get_preprocessor()
            
        # Manually encode geo columns to preserve them for embedding layers
        # This is done outside ColumnTransformer to keep the dataframe structure
        self.full_train = self.encode_geo_cols(self.full_train)
        self.test_values = self.encode_geo_cols(self.test_values)
        return self.full_train, self.test_values

    def get_geo_data(self):
        # Helper to get just geo columns for embedding training
        train_geo = self.full_train[self.geo_cols + ['damage_grade', 'building_id']]
        test_geo = self.test_values[self.geo_cols + ['building_id']]
        return train_geo, test_geo

    def encode_geo_cols(self, df):
        # Simple label encoding for geo columns to be used in embedding layers
        df_encoded = df.copy()
        for col in self.geo_cols:
            le = LabelEncoder()
            # Fit on all available data to cover all regions
            all_values = pd.concat([self.train_values[col], self.test_values[col]])
            le.fit(all_values)
            df_encoded[f"{col}_enc"] = le.transform(df[col])
        return df_encoded
