### insurance_fraud_detection

To run the application, follow the steps.
1. open the vscode terminal or cmd routing the project root folder.
2. run python -m src.data_ingestion to ingest the data
3. run python -m src.data_preprocessing to preprocess the data
4. run python -m src.feature_engineering for feature engineering
5. run python -m src.model_training to train the model
6. run python -m src.predictions to see the model predictions on the test data.
7. finally to check API, run uvicorn api.main:app --port 8080 --reload
8. it should open up a local URL - http://127.0.0.1:8080/
9. COpy and paste it in browser, it opens the swagger UI, open http://127.0.0.1:8084/docs, you will see the API methods created.
10. test the predict method to see the model predictions. 
