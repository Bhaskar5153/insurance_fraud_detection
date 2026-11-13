import joblib
import pandas as pd
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from pathlib import Path

class FraudPrediction:
    def __init__(self, model_path="artifacts/models/xgboost_pipeline.joblib"):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found:{self.model_path}")

        self.model = joblib.load(self.model_path)


    def predict(self, X, threshold=0.5):
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        return pd.DataFrame(
            {
                "predicted_fraud": predictions,
                "fraud_score": probabilities.round(4),
                "risk_level": [
                    "High" if p >= threshold else "Low" for p in probabilities
                ]
            }, index=X.index
        )


    def run(self):
        df = DataIngestion().load_data()
        processor = DataPreprocessing(df)
        X, y = processor.prepare_data()
        X_train, X_valid, X_test, y_train, y_valid, y_test = processor.split_data(X, y)


        X_sample = X_test.head(10).copy()
        y_sample = y_test.head(10)

        result = self.predict(X=X_sample)
        result.insert(0, "actual_fraud", y_sample.to_numpy())
        result.insert(1, "Correct", (result["actual_fraud"] == result["predicted_fraud"]))

        print("\nTest predictions")
        print(result.to_string(index=False))

        output = Path("artificats/reports/test_predictions.csv")
        output.parent.mkdir(parents=True, exist_ok=True)

        result.to_csv(output, index=False)
        print(f"\nPredictions saved: {output}")

        return result

if __name__ == "__main__":
    predictor = FraudPrediction()
    predictor.run()