from fastapi import FastAPI
from pydantic import BaseModel
from app.model.inference import predict_sentiment
from app.model.inference import __version__ as model_version

app = FastAPI()

class TextInput(BaseModel):
    text: str
    
class PredictionOutput(BaseModel):
    sentiment: str
    probability: float
    
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API", "model_version": model_version}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    sentiment, probability = predict_sentiment(input_data.text)
    return {"sentiment": sentiment, "probability": probability}

# Example: curl -X 'POST' \
#   'http://localhost/predict' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "text": "O produto chegou no prazo, estou satisfeito com a minha compra."
# }'