# app/model.py
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from efn.utils import load_and_preprocess_data, tokenize_data

class FinanceModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def train(self, train_texts, train_labels, val_texts, val_labels):
        train_encodings = tokenize_data(train_texts, self.tokenizer)
        val_encodings = tokenize_data(val_texts, self.tokenizer)
        
        # Convert labels to tensors
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        
        # Trainer configuration
        training_args = TrainingArguments(
            output_dir='./models/finance_model',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=(train_encodings, train_labels),
            eval_dataset=(val_encodings, val_labels),
            compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))}
        )
        
        trainer.train()
        self.model.save_pretrained('./models/fine_tuned_model')
    
    def predict(self, question):
        # Encode and predict
        inputs = self.tokenizer(question, return_tensors="pt")
        outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()
