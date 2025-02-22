# app/utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_and_preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Combine newssource and newscontent for simplicity
    data['text'] = data['newssource'] + ": " + data['newscontent']
    
    # Split dataset into training and test sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42
    )
    
    return train_texts, val_texts, train_labels, val_labels

def tokenize_data(texts, tokenizer, max_length=512):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
