# Customer Churn Prediction

A machine learning project focused on predicting customer churn using an optimized Random Forest classifier.

The project follows an end-to-end machine learning workflow, including data exploration, preprocessing, model development, hyperparameter tuning, evaluation, and model persistence.

---

## Project Overview

Customer churn is a major challenge for businesses that rely on customer retention.

The objective of this project is to develop a machine learning model that predicts whether a customer is likely to leave a service based on demographic information, subscription details, payment methods, and customer behavior.

This project uses a Random Forest classifier to identify potential churn patterns and evaluate predictive performance.

## Dataset

The dataset contains 500 customer records and 8 features.

| Feature | Description | Type |
|---|---|---|
| Age | Customer age | Numerical |
| Tenure | Duration of customer relationship | Numerical |
| MonthlyCharges | Monthly service charges | Numerical |
| SupportCalls | Number of customer support calls | Numerical |
| ContractType | Customer contract type | Categorical |
| InternetService | Type of internet service | Categorical |
| PaymentMethod | Customer payment method | Categorical |
| Churn | Customer churn status | Target |

### Target Distribution

- Yes: 407 customers (81.4%)
- No: 93 customers (18.6%)

The dataset contains missing values in `InternetService`, which are handled during preprocessing.

## Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy
- Joblib

## Machine Learning Workflow

The project follows these steps:

1. **Exploratory Data Analysis (EDA)**
   - Dataset inspection
   - Missing value analysis
   - Feature distribution visualization
   - Target variable analysis

2. **Data Preprocessing**
   - Median imputation for numerical features
   - Most frequent imputation for categorical features
   - One-Hot Encoding
   - ColumnTransformer and Pipeline

3. **Train-Test Split**
   - 80% Training Data
   - 20% Testing Data
   - Stratified sampling

4. **Model Development**
   - Random Forest Classifier
   - Baseline model evaluation
   - Stratified K-Fold Cross Validation

5. **Handling Class Imbalance**
   - Class weight experiments
   - Macro F1-score comparison

6. **Hyperparameter Tuning**
   - RandomizedSearchCV
   - Optimization using Macro F1-score

7. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

8. **Model Persistence**
   - Saving and loading the trained pipeline using Joblib

## Model Performance

The final Random Forest model achieved the following results on the test dataset:

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

The model demonstrates strong performance in identifying customers likely to churn, achieving a recall of 93% for the `Yes` class.

However, identifying customers who do not churn remains more challenging.

## Feature Importance

The Random Forest model identified the following features as the most influential:

| Feature | Importance |
|---|---:|
| MonthlyCharges | 44.30% |
| SupportCalls | 22.99% |
| Tenure | 13.49% |
| Age | 8.07% |

MonthlyCharges and SupportCalls were the most influential features in the model's predictions.

*Note: Feature importance represents the model's internal predictive importance estimates and does not establish causal relationships.*

## Installation

Clone the repository:

```bash
git clone https://github.com/Mobin-Madadi/customer-churn-decision-tree.git
```

Navigate to the project directory:

```bash
cd customer-churn-decision-tree
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib jupyter
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells sequentially.

The trained model can also be loaded using Joblib:

```python
import joblib

model = joblib.load("customer_churn_model.pkl")

predictions = model.predict(new_customer_data)
```

`new_customer_data` must contain the same input features used during training.

## Project Structure

```text
customer-churn-decision-tree/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── models/
│   └── customer_churn_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

*The directory structure may vary depending on the repository organization.*

## Limitations

- The dataset contains only 500 records.
- The target variable is imbalanced.
- Performance may vary on larger and more diverse datasets.
- Feature importance does not imply causation.
- Further validation is needed before using the model in a real business environment.

## Future Improvements

Potential future developments include:

- Testing additional machine learning algorithms.
- Advanced feature engineering.
- Model explainability using SHAP.
- Model deployment using FastAPI.
- Building a web-based prediction interface.
- Evaluating performance on larger datasets.

## Author

**Mobin Madadi**

Aspiring Machine Learning Engineer interested in Artificial Intelligence, Machine Learning, and Generative AI.

GitHub: [Mobin-Madadi](https://github.com/Mobin-Madadi)

---

*This project was developed as part of my practical machine learning learning journey.*
