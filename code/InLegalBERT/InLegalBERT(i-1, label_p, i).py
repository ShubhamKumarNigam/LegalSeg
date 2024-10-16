import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import pandas as pd
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parameters
PRETRAINED_MODEL_NAME = 'law-ai/InLegalBERT' 
NUM_CLASSES = 7  # Number of rhetorical roles
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1  # Process one document at a time
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# Label mapping (adjust according to your dataset)
label_id_to_text = {
    0: 'None',
    1: 'Facts',
    2: 'Issue',
    3: 'Argument by Petitioner',
    4: 'Argument by Respondent',
    5: 'Reasoning',
    6: 'Decision',
}

label_text_to_id = {v: k for k, v in label_id_to_text.items()}

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

# Custom Dataset
class LegalDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.documents = dataframe['index'].unique()
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        for doc_id in self.documents:
            doc_df = self.dataframe[self.dataframe['index'] == doc_id]
            sentences = doc_df['text'].tolist()
            labels = doc_df['label'].tolist()
            samples.append({
                'index': doc_id,
                'text': sentences,
                'label': labels
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Define the model
class BertForSentenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(BertForSentenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load data from CSV files
train_df = pd.read_csv('../../data/InLegalBERT/train.csv')
val_df = pd.read_csv('../../data/InLegalBERT/data/val.csv')
test_df = pd.read_csv('../../data/InLegalBERT/test.csv')

# Data preprocessing
def preprocess_df(df):
    df = df.dropna(subset=['text', 'label'])  # Drop rows with missing values
    df['label'] = df['label'].astype(int)
    return df

train_df = preprocess_df(train_df)
val_df = preprocess_df(val_df)
test_df = preprocess_df(test_df)

# Create datasets
train_dataset = LegalDataset(train_df)
val_dataset = LegalDataset(val_df)
test_dataset = LegalDataset(test_df)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = BertForSentenceClassification(PRETRAINED_MODEL_NAME, NUM_CLASSES)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Training and evaluation functions
def train():
    model.train()
    total_loss = 0
    total_sentences = 0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        # Since batch size is 1, batch is the sample itself
        doc = batch
        sentences = doc['text']
        labels = doc['label']
        num_sentences = len(sentences)
        total_sentences += num_sentences

        prev_sentence = "[CLS]"  # Placeholder for the first sentence
        prev_label = "None"  # Initialize previous predicted label

        for i in range(num_sentences):
            current_sentence = sentences[i]

            # Construct the input text
            input_text = (
                f"[PREV_SENT] {prev_sentence} [PREV_LABEL] {prev_label} [CUR_SENT] {current_sentence}"
            )

            encoding = tokenizer(
                input_text,
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            label = torch.tensor([labels[i]], dtype=torch.long).to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, label)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update prev_sentence and prev_label with predicted label
            preds = torch.argmax(logits, dim=1)
            prev_label = label_id_to_text[int(preds.cpu().numpy()[0])]
            prev_sentence = current_sentence

        # Update model parameters after processing the document
        optimizer.step()
        # Gradients are already zeroed at the start of the next loop

        avg_loss = total_loss / total_sentences
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})

    avg_loss = total_loss / total_sentences
    return avg_loss

def evaluate(loader, phase='Validation'):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    progress_bar = tqdm(loader, desc=f'{phase}')
    with torch.no_grad():
        for batch in progress_bar:
            # Since batch size is 1, batch is the sample itself
            doc = batch  # Corrected line
            sentences = doc['text']
            labels = doc['label']
            num_sentences = len(sentences)

            prev_sentence = "[CLS]"  # Placeholder for the first sentence
            prev_label = "None"  # Initialize previous predicted label

            for i in range(num_sentences):
                current_sentence = sentences[i]

                # Construct the input text
                input_text = (
                    f"[PREV_SENT] {prev_sentence} [PREV_LABEL] {prev_label} [CUR_SENT] {current_sentence}"
                )

                encoding = tokenizer(
                    input_text,
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    return_tensors='pt',
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                label = torch.tensor([labels[i]], dtype=torch.long).to(device)

                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, label)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                # Update prev_sentence and prev_label with predicted label
                prev_label = label_id_to_text[int(preds.cpu().numpy()[0])]
                prev_sentence = current_sentence

                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    avg_loss = total_loss / len(all_labels)
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    mcc = matthews_corrcoef(all_labels, all_preds)
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mcc': mcc,
    }
    df = pd.DataFrame({
    'true': all_labels,
    'pred': all_preds
    })
    df.to_csv('predictions.csv')
    return metrics


# Training loop
best_val_f1 = 0
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    train_loss = train()
    val_metrics = evaluate(val_loader, phase='Validation')
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Metrics: "
          f"Loss: {val_metrics['loss']:.4f}, "
          f"Accuracy: {val_metrics['accuracy']:.4f}, "
          f"Precision: {val_metrics['precision']:.4f}, "
          f"Recall: {val_metrics['recall']:.4f}, "
          f"F1 Score: {val_metrics['f1_score']:.4f}, "
          f"MCC: {val_metrics['mcc']:.4f}")
    # Save the best model
    if val_metrics['f1_score'] > best_val_f1:
        best_val_f1 = val_metrics['f1_score']
        torch.save(model.state_dict(), 'best_model.pt')

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate on test set
test_metrics = evaluate(test_loader, phase='Testing')
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1_score']:.4f}")
print(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}")
