from pydantic import BaseModel , Field


class CustomerData(BaseModel):
    Age: int = Field(... , ge = 18 , le = 100)
    Tenure: int = Field(... , ge = 0)
    MonthlyCharges: float = Field(... , ge = 0)
    SupportCalls: int = Field(... , ge = 0)

    ContractType: str
    InternetService: str | None = None
    PaymentMethod: str



class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    retention_probability: float