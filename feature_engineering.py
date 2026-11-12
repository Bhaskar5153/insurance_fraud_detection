import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, quick_submission_days=5):
        self.quick_submission_days = quick_submission_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        required = [
            "Claim_Amount", 
            "Length_of_Stay",
            "Days_Between_Service_and_Claim",
            "Claim_Submission_Date",
            "Number_of_Claims_Per_Provider_Monthly",
            "Prior_Visits_12m",

        ]

        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")

        numerical_columns = [
            "Claim_Amount", 
            "Length_of_Stay",
            "Days_Between_Service_and_Claim",
            "Number_of_Claims_Per_Provider_Monthly",
            "Prior_Visits_12m"
        ]

        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

        dates = pd.to_datetime(
            df["Claim_Submission_Date"], errors="coerce"
        )

        # 1 . log transformation of claim amount

        df["log_Claim_Amount"] = np.log1p(df['Claim_Amount'])

        # 2. claims duration to complete
        df["Claim_Amount_Per_Stay"] = (df["Claim_Amount"] / (df["Length_of_Stay"] + 1))

        # 3. quick sumbission indicator
        delay = df["Days_Between_Service_and_Claim"]
        df["Quick_Submission"] = (delay <= self.quick_submission_days).astype(float)
        df.loc[delay.isna(), "Quick_Submission"] = np.nan


        # 4. Submission Date

        df["Submission_Month"] = dates.dt.month
        df["Submission_DayOfWeek"] = dates.dt.dayofweek
        df["Submission_IsWeekend"] = (dates.dt.dayofweek >= 5).astype(float)
        df.loc[dates.isna(), "Submision_IsWeekend"] = np.nan


        df["Provider_Claim_Volume_Log"] = np.log1p(df["Number_of_Claims_Per_Provider_Monthly"])

        df["Claim_Amount_Per_Prior_Visit"] = (df["Claim_Amount"] / df["Prior_Visits_12m"] + 1)

        return df


if __name__ == "__main__":
    df = DataIngestion().load_data()
    processor = DataPreprocessing()

    X, y = processor.prepare_data()

    engineering = FeatureEngineering()

    X_engineered = engineering.fit_transform(X)

    new_features = sorted(set(X_engineered.columns))

    print(X_engineered[new_features].head())