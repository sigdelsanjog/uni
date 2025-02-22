import torch
from models.transformer import TransformerModel
from models.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer()
tokenizer.load_vocab("saved_models/tokenizer.json")

# Load model
VOCAB_SIZE = len(tokenizer.word2idx)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size=VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("saved_models/transformer.pth", map_location=DEVICE))
model.eval()

def generate_response(prompt, max_length=50):
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor, input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_tensor.squeeze().tolist())
