from pathlib import Path

import joblib
import pandas as pd

# Get Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Model Path
MODEL_PATH = BASE_DIR / 'model' / 'customer_churn_model.pkl'

# Load trained model
model = joblib.load(MODEL_PATH)

def predict_churn(customer_data: dict):
    """

    Predict customer churn using the trained ML pipline.
    """

    # Convert input data to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Make prediction
    prediction = model.predict(customer_df)[0]

    # Get prediction probabilities 
    probabilities = model.predict_proba(customer_df)[0]

    # Map class Labels to probabilities
    probability_map = dict(zip(model.class_ , probabilities))

    return {
        'prediction': str(prediction),
        'churn_probability': float(probability_map.get('Yes', 0)),
        'retention_probability': float(probability_map.get('No' , 0))
    }