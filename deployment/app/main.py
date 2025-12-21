from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.model.inference import predict_sentiment
from app.model.inference import __version__ as model_version

app = FastAPI()

# Serve static files and templates for a simple frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


class TextInput(BaseModel):
    text: str
    

class PredictionOutput(BaseModel):
    sentiment: str
    probability: float


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_version": model_version})


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    """JSON API endpoint used by the frontend JavaScript to get predictions."""
    sentiment, probability = predict_sentiment(input_data.text)
    return {"sentiment": sentiment, "probability": probability}