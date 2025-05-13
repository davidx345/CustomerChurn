<<<<<<< HEAD
# CustomerChurn
A machine learning project to determine when customers will most likely leave a company . I made it for a bank situation
=======
# Bank Customer Churn Prediction

This project predicts whether a bank customer will churn using a machine learning model trained on the Kaggle dataset: [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).

## Features
- Python backend (Flask)
- Machine learning model (Random Forest, scikit-learn)
- Preprocessing: Label Encoding, StandardScaler
- API endpoint for predictions (`/predict`)
- Simple web UI (HTML/CSS/JS)
- Model evaluation and feature importance visualization

## File Structure
- `train_model.py` — Train and save the model (`churn_model.pkl`)
- `predict_churn.py` — Input validation and prediction logic
- `app.py` — Flask API and web server
- `templates/index.html` — Web form UI
- `static/style.css` — Styling for the form
- `feature_importance.png` — Feature importance plot (generated after training)
- `churn_model.pkl` — Saved model and preprocessors

## Setup & Usage

### 1. Install Requirements
```
pip install -r requirements.txt
```
Or manually:
```
pip install flask pandas scikit-learn joblib matplotlib seaborn
```

### 2. Download Dataset
Download `Bank Customer Churn Prediction.csv` from Kaggle and place it in the project root.

### 3. Train the Model
```
python train_model.py
```
This will output `churn_model.pkl` and `feature_importance.png`.

### 4. Run the Web App
```
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

### 5. API Usage
Send a POST request to `/predict` with JSON body:
```
{
  "CreditScore": 600,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```
Response:
```
{
  "prediction": "No",
  "probability": 0.13
}
```

## Notes
- All preprocessing is handled automatically.
- For retraining, rerun `train_model.py` after updating the dataset.
- For feature importance, see `feature_importance.png`.

## License
MIT
>>>>>>> 8b69ad5 (Initial commit: customer churn prediction project)
