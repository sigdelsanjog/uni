import json

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}  # PAD and UNK tokens
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        """Builds vocabulary from a list of texts"""
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, text, max_length=None):
        """Encodes text into token indices with optional padding"""
        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]
        
        if max_length:
            tokens = tokens[:max_length]  # Truncate if too long
            tokens += [self.word2idx["<PAD>"]] * (max_length - len(tokens))  # Pad if too short
        
        return tokens

    def decode(self, token_ids):
        """Decodes token indices back to text"""
        return " ".join(self.idx2word.get(idx, "<UNK>") for idx in token_ids)

    def save_vocab(self, filepath):
        """Saves vocabulary to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.word2idx, f)

    def load_vocab(self, filepath):
        """Loads vocabulary from a JSON file"""
        with open(filepath, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
