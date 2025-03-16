import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

with zipfile.ZipFile('train_data.zip') as zf:
    data = pd.read_csv(zf.open('realty_data.csv'))


data.dropna(subset=['total_square', 'rooms', 'price'], inplace=True)

X = data[['total_square', 'rooms']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'elastic_model.pkl')
app = FastAPI()

with open("elastic_model.pkl", 'rb') as file:
    model = joblib.load(file)

class ModelRequestData(BaseModel):
    total_square: float
    rooms: float

class Result(BaseModel):
    result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.get("/predict_get")
def predict_get(total_square: float, rooms: float):
    input_data = {
        "total_square": total_square,
        "rooms": rooms
    }
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

