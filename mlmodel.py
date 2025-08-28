from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Sample data for ML training
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

# In-memory data storage
stored_data = [{"input": 1, "prediction": 2}]

# Request body model
class InputData(BaseModel):
    input_value: float

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.get("/data")
def get_data():
    return {"data": stored_data}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.input_value]])[0]
    result = {"input": data.input_value, "prediction": prediction}
    stored_data.append(result)
    return result
