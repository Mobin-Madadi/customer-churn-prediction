# Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn using a Scikit-learn classification pipeline, exposed through a REST API built with FastAPI.

## Project Overview

Customer churn refers to customers discontinuing their relationship with a company or service.

The goal of this project is to develop a classification model that identifies customers who are more likely to leave, based on their demographic information, service usage, and subscription details.

The trained model is deployed through a FastAPI application, allowing users to submit customer information and receive churn predictions with probability estimates.

## Key Features

- Exploratory Data Analysis (EDA)
- Data cleaning and missing-value handling
- Categorical feature encoding
- Random Forest classification
- Cross-validation and hyperparameter tuning
- Model evaluation and feature importance analysis
- Model persistence using Joblib
- REST API development with FastAPI
- Input validation using Pydantic
- Automated API testing with Pytest

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model persistence |
| FastAPI | REST API |
| Pydantic | Input validation |
| Uvicorn | ASGI server |
| Pytest | Automated testing |

## Dataset

The dataset contains 500 customer records and 8 columns.

### Features

| Feature | Description |
|---|---|
| Age | Customer age |
| Tenure | Duration of customer relationship |
| MonthlyCharges | Monthly service charges |
| SupportCalls | Number of customer support calls |
| ContractType | Customer contract type |
| InternetService | Internet service type |
| PaymentMethod | Customer payment method |
| Churn | Target variable |

The target variable has two classes:

- `Yes`: Customer churned.
- `No`: Customer retained.

Missing values in categorical features are handled through preprocessing.

## Machine Learning Workflow

1. Data loading and exploration
2. Exploratory Data Analysis
3. Missing-value handling
4. Numerical feature imputation
5. Categorical feature imputation and One-Hot Encoding
6. Train-test split
7. Random Forest model development
8. Cross-validation
9. Class imbalance handling
10. Hyperparameter tuning
11. Model evaluation
12. Feature importance analysis
13. Model serialization using Joblib

The preprocessing and classification steps are combined into a Scikit-learn Pipeline to ensure consistent transformations during training and inference.

## Model Performance

The final model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 86% |
| Macro F1-score | 0.76 |
| Weighted F1-score | 0.86 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No | 0.65 | 0.58 | 0.61 |
| Yes | 0.90 | 0.93 | 0.91 |

These results indicate that the model performs better at identifying customers who churn than customers who stay.

### Model Visualizations

#### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

#### Feature Importance

![Feature Importance](images/feature_importance.png)

## REST API

The trained model is accessible through a REST API developed with FastAPI.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Customer churn prediction |

### Example Request

`POST /predict`

```json
{
  "Age": 35,
  "Tenure": 12,
  "MonthlyCharges": 85.5,
  "SupportCalls": 4,
  "ContractType": "Monthly",
  "InternetService": "Fiber",
  "PaymentMethod": "Electronic Check"
}
```

### Example Response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.9046,
  "retention_probability": 0.0954
}
```

The response probabilities are model estimates and should not be interpreted as guaranteed real-world outcomes.

### Interactive API Documentation

FastAPI automatically generates interactive API documentation.

Once the application is running, visit:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Mobin-Madadi/customer-churn-decision-tree.git
cd customer-churn-decision-tree
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

## Running Tests

Run the automated test suite:

```bash
PYTHONPATH=. python -m pytest -v
```

The tests cover:

- Home endpoint
- Health endpoint
- Prediction endpoint
- Invalid input validation

## Project Structure

```text
customer-churn-decision-tree/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── data/
│   └── customer_churn.csv
│
├── model/
│   └── customer_churn_model.pkl
│
├── notebooks/
│   ├── customer_churn.csv
│   └── customer_churn.ipynb
│
├── tests/
│   └── test_api.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations & Future Improvements

Potential improvements include:

- Probability calibration
- More extensive model comparison
- Additional customer behavior features
- Improved class imbalance handling
- Docker containerization
- Cloud deployment
- CI/CD integration
- Monitoring model performance over time

## Author

**Mobin Madadi**

GitHub: [Mobin-Madadi](https://github.com/Mobin-Madadi)

This project was developed as part of my practical learning journey in Machine Learning and AI Engineering.
