from fastapi import FastAPI
from app.schemas import CustomerData , PredictionResponse
from app.model import predict_churn


app = FastAPI(
    title = 'Customer Churn Prediction API',
    description = 'An ML-powered API for predicting customer churn',
    version = '1.0.0'
)


@app.get('/')
def home():
    return {
        'message': 'Customer Churn Prediction API',
        'status': 'running'
    }


@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': True
    }


@app.post('/predict' , response_model = PredictionResponse)
def predict(customer: CustomerData):
    customer_data = customer.model_dump()

    resualt = predict_churn(customer_data)

    return resualt