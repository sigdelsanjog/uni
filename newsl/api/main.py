from fastapi import FastAPI
from api.inference import generate_response

app = FastAPI()

@app.post("/ask")
async def ask_question(prompt: str):
    response = generate_response(prompt)
    return {"response": response}

# Run the server using: uvicorn api.main:app --reload
