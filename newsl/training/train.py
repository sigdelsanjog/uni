import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerModel
from training.dataset import get_dataloader
from models.tokenizer import Tokenizer
import pandas as pd
import sys
import os

# Ensure the correct path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
tokenizer = Tokenizer()
df = pd.read_csv("data/english.csv")

# Build vocabulary
tokenizer.build_vocab(df["newscontents"].tolist())

# Hyperparameters
VOCAB_SIZE = len(tokenizer.word2idx)
EMBEDDING_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 512
NUM_LAYERS = 6
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 128  # Ensures consistent input length

# Initialize Model, Loss, Optimizer
model = TransformerModel(
    vocab_size=VOCAB_SIZE, 
    embed_dim=EMBEDDING_DIM, 
    num_heads=NUM_HEADS, 
    hidden_dim=HIDDEN_DIM, 
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load data
dataloader = get_dataloader("data/english.csv", tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LEN)

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, tgt)

        # Reshape output to match target dimensions
        loss = criterion(output.view(-1, VOCAB_SIZE), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Save the model and tokenizer
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/transformer.pth")
tokenizer.save_vocab("saved_models/tokenizer.json")

print("Training complete. Model and tokenizer saved.")
