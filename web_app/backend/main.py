from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import logging
from model import TransformerModel, PositionalEncoding


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware to allow requests from localhost:8080
origins = [
    "http://localhost:8080",  # Allow localhost:8080
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model
model_path = 'model_files/my_llm_model.keras'  # Update this path if necessary
logging.info(f"Loading model from {model_path}")

# Create FastAPI app
app = FastAPI()

class InputData(BaseModel):
    input_text: str

# Custom object scope for model loading
with tf.keras.utils.custom_object_scope({
    "PositionalEncoding": PositionalEncoding,
    "TransformerModel": TransformerModel
}):
    model = tf.keras.models.load_model(model_path)

@app.post("/predict")
async def predict(input_data: InputData):
    # Perform prediction
    input_text = input_data.input_text
    # Preprocess input_text as needed
    # Example: Convert text to tokens
    tokens = preprocess_input(input_text)  
    predictions = model.predict(tokens)
    output_text = postprocess_predictions(predictions)  # Optional based on your use case

    
    # For demonstration, return the same input
    return {"output": f"Processed: {input_text}"}
