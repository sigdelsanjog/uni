# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from efn.model import FinanceModel
import torch

app = FastAPI()

# Load the model and tokenizer
model = FinanceModel()
model.model.load_state_dict(torch.load("./models/fine_tuned_model/pytorch_model.bin"))
model.model.eval()

class Question(BaseModel):
    question: str

@app.post("/train")
async def train_model():
    try:
        train_texts, val_texts, train_labels, val_labels = load_and_preprocess_data('./data/finance_news.csv')
        model.train(train_texts, train_labels, val_texts, val_labels)
        return {"message": "Model training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_answer(question: Question):
    try:
        answer_label = model.predict(question.question)
        return {"label": answer_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
