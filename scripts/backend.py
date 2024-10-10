from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json

# Load the trained model
model = tf.keras.models.load_model('../uni/models/my_llm_model.h5')

# Load the token mapping from tokens.json
with open('../uni/data/tokens.json', 'r') as f:
    tokens_data = json.load(f)

# Initialize the tokens and reverse mapping
tokens = {}
reverse_tokens = {}

# Create a single index for all tokens across faculties
index = 0
for faculty, token_list in tokens_data.items():
    for token in token_list:
        if token not in tokens:  # Only add unique tokens
            tokens[token] = index
            reverse_tokens[index] = token
            index += 1

app = FastAPI()

# Define the input data model using Pydantic
class InputData(BaseModel):
    input: str

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Get the input string from the request
        input_string = data.input

        # Convert the input string to tokens
        input_tokens = [tokens.get(char, 0) for char in input_string]

        # Pad or truncate the input tokens to match the model's expected input shape
        max_length = 100  # Example max length; should match the model's training max length
        input_tokens = input_tokens[:max_length] + [0] * (max_length - len(input_tokens))

        # Convert input tokens to a numpy array for prediction
        input_data = np.array([input_tokens])  

        # Use the model to predict
        predictions = model.predict(input_data)

        # Get the predicted class indices for each time step
        predicted_indices = np.argmax(predictions, axis=-1)

        # Convert indices back to human-readable tokens
        predicted_tokens = [reverse_tokens.get(idx, "Unknown token") for idx in predicted_indices[0]]

        response = {
            "predicted_tokens": predicted_tokens,
            "predictions": predictions.tolist()  # You can keep this if needed for debugging
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
