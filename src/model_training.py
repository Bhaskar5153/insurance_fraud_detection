import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering


class ModelTraining:
    def __init__(self):
        self.model_dir = Path("artifacts/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self, y_train):
        fraud = (y_train == 1).sum()
        non_fraud = (y_train == 0).sum()
        if fraud == 0 or non_fraud == 0:
            raise ValueError("Training data should have both fraud and non fraud classes")

        scale_pos_weight = non_fraud / fraud
        print(f"Scale Pos Weight: {scale_pos_weight}")


        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.5,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            objective="binary:logistic",
            colsample_bytree=0.9

        )


    def build_pipeline(self, processor, model):
        return Pipeline(
            [
                ("feature_engineering", FeatureEngineering()),
                ("preprocessing", processor.build_preprocessor()),
                ("model", model)
            ]
        )

    def train(self):
        print("Starting XGBoost model training.....")

        # step 1
        # get the data from SQL
        df = DataIngestion().load_data()

        # step 2 
        # prepare the dataset, target and features
        processor = DataPreprocessing(df)
        X, y = processor.prepare_data()

        # step 3
        X_train, X_valid, X_test, y_train, y_valid, y_test = processor.split_data(X, y)

        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")

        # step 4
        model = self.build_model(y_train=y_train)
        pipeline = self.build_pipeline(processor=processor, model=model)

        # step 5

        pipeline.fit(X_train, y_train)
        print("XGBoost training completed")

        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        if y_valid.nunique() < 2:
            raise ValueError("Validation dataset must contain both 0 and 1 classes")

        pr_auc = average_precision_score(y_valid, probabilities)
        roc_auc = roc_auc_score(y_valid, probabilities)

        print(f"Validation PR-AUC: {pr_auc:.4f}")
        print(f"Validation ROC-AUC: {roc_auc:.4f}")


        # step 7
        model_path = self.model_dir / "xgboost_pipeline.joblib"
        joblib.dump(pipeline, model_path)

        # step 8 - save the metadata
        metadata = {
            "model": "XGBoost",
            "algorithm": "XGBClassifier",
            "training_records": len(X_train),
            "validation_ records": len(X_valid),
            "test_records": len(X_test),
            "scale_pos_weight": float(model.scale_pos_weight),
            "validation_pr_auc": float(pr_auc),
            "validation_roc_auc": float(roc_auc),
            "test_evaluated": False

        }

        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model saved: {model_path}")

        return pipeline, X_test, y_test



if __name__ == "__main__":
    trainer = ModelTraining()
    pipeline, X_test, y_test = trainer.train()

    

        

        
        


