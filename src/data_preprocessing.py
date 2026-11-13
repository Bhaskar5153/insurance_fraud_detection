import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data_ingestion import DataIngestion

import ipdb

class DataPreprocessing:
    TARGET = "Is_Fraud"
    DATE_COLUMN = "Claim_Submission_Date"
    DROP_COLUMNS = [
        "Provider_ID",
        "Claim_ID",
        "Approved_Amount",
        "Patient_Gender",
        "Claim_Status",
        "Patient_State"
    ]

    NUMERICAL_COLUMNS = [
        "Patient_Age",
        "Claim_Amount",
        "Days_Between_Service_and_Claim",
        "Number_of_Claims_Per_Provider_Monthly",
        "Length_of_Stay",
        "Prior_Visits_12m",
        "Procedure_Code",
        "Chronic_Condition_Flag"

    ]

    CATEGORICAL_COLUMNS = [
        "Diagnosis_Code",
        "Insurance_Type",
        "Provider_Specialty",
        "Visit_Type",
        
    ]



    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.preprocessor = None


    def prepare_data(self):
        df = self.df.copy()
        required = (self.NUMERICAL_COLUMNS + self.CATEGORICAL_COLUMNS + [self.TARGET, self.DATE_COLUMN])

        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df[self.DATE_COLUMN] = pd.to_datetime(df[self.DATE_COLUMN], errors="coerce")

        if df[self.DATE_COLUMN].isna().any():
            raise ValueError("Claim_Submission_Date has missing dates")

        y = pd.to_numeric(df[self.TARGET], errors="coerce")
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError("Is_Fraud must have only 0 and 1")


        for col in self.NUMERICAL_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df[self.CATEGORICAL_COLUMNS] = (df[self.CATEGORICAL_COLUMNS].astype("string"))


        X = df.drop(columns=[self.TARGET] + self.DROP_COLUMNS)
        y = y.astype(int)

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        if len(X) < 20:
            raise ValueError("Insufficeient records to split the data")

        order = X.sort_values(self.DATE_COLUMN, kind="stable").index

        X = X.loc[order].reset_index(drop=True)
        y = y.loc[order].reset_index(drop=True)

        train_end = int(len(X) * 0.70)
        valid_end = int(len(X) * 0.85)

        X_train = X.iloc[:train_end].copy()
        X_valid = X.iloc[train_end:valid_end].copy()
        X_test = X.iloc[valid_end:].copy()


        y_train = y.iloc[:train_end].copy()
        y_valid = y.iloc[train_end:valid_end].copy()
        y_test = y.iloc[valid_end:].copy()

        return X_train, X_valid, X_test, y_train, y_valid, y_test


    def build_preprocessor(self):
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
            ]
        )


        self.preprocessor = ColumnTransformer(
            [
                ("numerical", numerical_pipeline, self.NUMERICAL_COLUMNS),
                ("categorical", categorical_pipeline, self.CATEGORICAL_COLUMNS)
            ]
        )
        return self.preprocessor


    def run(self):
        print("starting data preprocessing.....")
        X, y = self.prepare_data()

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.split_data(X, y)

        preprocessor = self.build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_valid_processed = preprocessor.fit_transform(X_valid)
        X_test_processed = preprocessor.fit_transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        for idx, feature_name in enumerate(feature_names):
            print(f"{idx}: {feature_name}")
        ipdb.set_trace()

        print(X_train_processed.shape)
        print(X_valid_processed.shape)
        print(X_test_processed.shape)
        print(X_train_processed.ndim)

        return {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
            "processor": preprocessor
        }
    

if __name__ == "__main__":
    df = DataIngestion().load_data()
    processor = DataPreprocessing(df)
    result = processor.run()

    





        





