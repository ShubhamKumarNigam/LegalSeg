from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import pandas as pd
import numpy as np
import os

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parameters
PRETRAINED_MODEL_NAME = 'law-ai/InLegalBERT'
NUM_CLASSES = 7  # Number of rhetorical roles
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# Initialize tokenizer and BERT model
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
bert_model.to(device)
bert_model.eval()  # Set BERT to evaluation mode

# Function to get sentence embeddings using BERT
def get_sentence_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt',
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        # Use [CLS] token representation
        embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
    return embedding.squeeze(0)  # (hidden_size,)

# Custom PyTorch Geometric dataset
class LegalGraphDataset(GeoDataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.documents = dataframe['Document_id'].unique()
        self.num_classes = NUM_CLASSES
        super(LegalGraphDataset, self).__init__()
    
    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

    def len(self):
        return len(self.documents)

    def get(self, idx):
        doc_id = self.documents[idx]
        doc_df = self.dataframe[self.dataframe['Document_id'] == doc_id]

        sentences = doc_df['sentences'].tolist()
        labels = doc_df['labels'].tolist()

        # Get embeddings for all sentences in the document
        embeddings = []
        for sentence in sentences:
            emb = get_sentence_embedding(sentence)
            embeddings.append(emb.cpu())

        x = torch.stack(embeddings)  # (num_nodes, hidden_size)
        y = torch.tensor(labels, dtype=torch.long)  # (num_nodes,)

        num_nodes = x.size(0)

        # Create edge index (connect sentences sequentially)
        edge_index = torch.tensor(
            [[], []], dtype=torch.long
        )  # Initialize empty edge index
        if num_nodes > 1:
            # Connect each sentence to the next one
            src = torch.arange(0, num_nodes - 1, dtype=torch.long)
            dst = torch.arange(1, num_nodes, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            # Add reverse edges
            rev_edge_index = torch.stack([dst, src], dim=0)
            edge_index = torch.cat([edge_index, rev_edge_index], dim=1)

        data = GraphData(x=x, edge_index=edge_index, y=y)
        return data

# Define the GNN model
class SentenceGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SentenceGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # x: [num_nodes, input_dim]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # No pooling; we want node-level outputs
        logits = self.classifier(x)  # [num_nodes, num_classes]
        return logits

# Load data from CSV files

def modify(train_df, val_df, test_df):
  mymap = {
      "Facts" : 0,
      "Issue" : 1,
      "Arguments of Petitioner" : 2,
      "Arguments of Respondent" : 3,
      "Reasoning" : 4,
      "Decision" : 5,
      "None" : 6
  }

  train_df = train_df.rename(columns={'Text': 'sentences', 'Label': 'labels', 'Index': 'Document_id'})
  val_df = val_df.rename(columns={'Text': 'sentences', 'Label': 'labels', 'Index': 'Document_id'})
  test_df = test_df.rename(columns={'Text': 'sentences', 'Label': 'labels', 'Index': 'Document_id'})

  train_df['labels'] = train_df['labels'].map(mymap)
  val_df['labels'] = val_df['labels'].map(mymap)
  test_df['labels'] = test_df['labels'].map(mymap)
  train_df['labels'].fillna(6, inplace=True)
  val_df['labels'].fillna(6, inplace=True)
  test_df['labels'].fillna(6, inplace=True)
  return train_df, val_df, test_df

train_df = pd.read_csv('../data/train.csv')
val_df = pd.read_csv('../data/val.csv')
test_df = pd.read_csv('../data/test.csv')

#train_df = train_df.head(train_df[train_df['Index']==100].index[-1])
#val_df = val_df.head(val_df[val_df['Index']==5004].index[-1])
#test_df = test_df.head(test_df[test_df['Index']==6718].index[-1])
train_df, val_df, test_df = modify(train_df, val_df, test_df)

# Data preprocessing
def preprocess_df(df):
    df = df.dropna(subset=['sentences', 'labels'])  # Drop rows with missing values
    df['labels'] = df['labels'].astype(int)
    return df

train_df = preprocess_df(train_df)
val_df = preprocess_df(val_df)
test_df = preprocess_df(test_df)

# Create datasets
train_dataset = LegalGraphDataset(train_df)
val_dataset = LegalGraphDataset(val_df)
test_dataset = LegalGraphDataset(test_df)

# Create dataloaders
train_loader = GeoDataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = GeoDataLoader(val_dataset, batch_size=1)
test_loader = GeoDataLoader(test_dataset, batch_size=1)

# Initialize model
hidden_dim = 128
input_dim = bert_model.config.hidden_size
model = SentenceGNN(input_dim, hidden_dim, NUM_CLASSES)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and evaluation functions
def train():
    model.train()
    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    with open('predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['all_labels', 'all_preds'])
        for gold, pred in zip(all_labels, all_preds):
            writer.writerow([labels, preds])
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    mcc = matthews_corrcoef(all_labels, all_preds)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mcc': mcc,
    }
    return metrics

# Training loop
best_val_f1 = 0
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train()
    val_metrics = evaluate(val_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Metrics: "
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
model.load_state_dict(torch.load('../../saved_models/GNN/best_model.pt'))

# Evaluate on test set
test_metrics = evaluate(test_loader)
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1_score']:.4f}")
print(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}")
