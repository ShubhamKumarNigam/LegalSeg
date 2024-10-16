from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import pandas as pd
import numpy as np
import os

def modify(train_df, val_df, test_df):
  mymap = {
      "Facts" : 0,
      "Issue" : 1,
      "Arguments by Petitioner" : 2,
      "Arguments by Respondent" : 3,
      "Reasoning" : 4,
      "Decision" : 5,
      "None" : 6
  }

  train_df = train_df.rename(columns={'Text': 'sentences', 'Label': 'labels'})
  val_df = val_df.rename(columns={'Text': 'sentences', 'Label': 'labels'})
  test_df = test_df.rename(columns={'Text': 'sentences', 'Label': 'labels'})

  train_df['labels'] = train_df['labels'].map(mymap)
  val_df['labels'] = val_df['labels'].map(mymap)
  test_df['labels'] = test_df['labels'].map(mymap)
  train_df['labels'].fillna(6, inplace=True)
  val_df['labels'].fillna(6, inplace=True)
  test_df['labels'].fillna(6, inplace=True)
  return train_df, val_df, test_df

train_df = pd.read_csv('../../data/Role-Aware/train.csv')
val_df = pd.read_csv('../../data/Role-Aware/val.csv')
test_df = pd.read_csv('../../data/Role-Aware/test.csv')
train_df, val_df, test_df = modify(train_df, val_df, test_df)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the Role-Aware Transformer model
class RoleAwareTransformer(nn.Module):
    def __init__(self, pretrained_model_name, num_roles):
        super(RoleAwareTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.role_embeddings = nn.Embedding(num_roles, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_roles)
        self.num_roles = num_roles

    def forward(self, input_ids, attention_mask, role_ids=None):
        # Get token embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Mean pooling over the sequence length to get sentence representation
        pooled_output = torch.mean(sequence_output, dim=1)  # (batch_size, hidden_size)

        if role_ids is not None:
            # Get role embeddings
            role_embedding = self.role_embeddings(role_ids)  # (batch_size, hidden_size)
            # Add role embeddings to the pooled output
            pooled_output += role_embedding

        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, num_roles)
        return logits

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss  # Since we want to minimize val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

# Custom dataset class
class LegalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.sentences = dataframe['sentences'].tolist()
        self.labels = dataframe['labels'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def train_model(model, dataloader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Processing batches"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_ids=labels
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def evaluate_model(model, dataloader, epoch=None, writer=None, phase='Validation'):
    model.eval()
    all_preds = []
    all_labels = []
    df = pd.DataFrame({'true': all_labels, 'pred': all_preds})
    df.to_csv('predictions.csv', index=False)
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                role_ids=None  # No role IDs during evaluation
            )
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
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

    if writer is not None and epoch is not None:
        writer.add_scalar(f'Loss/{phase}', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/Precision', precision, epoch)
        writer.add_scalar(f'{phase}/Recall', recall, epoch)
        writer.add_scalar(f'{phase}/F1_Score', f1_score, epoch)
        writer.add_scalar(f'{phase}/MCC', mcc, epoch)

    return metrics

if __name__ == "__main__":
    # Parameters
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'
    NUM_ROLES = 7  # Number of rhetorical roles
    MAX_SEQ_LENGTH = 512  # Adjusted to maximum allowed by BERT
    BATCH_SIZE = 4  # Reduced to handle larger sequence length
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5
    PATIENCE = 3  # For early stopping
    MODEL_SAVE_PATH = 'best_model.pt'

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

    # Data preprocessing
    def preprocess_df(df):
        df = df.dropna(subset=['sentences', 'labels'])  # Drop rows with missing values
        df['labels'] = df['labels'].astype(int)
        return df

    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)

    # Create datasets and dataloaders
    train_dataset = LegalDataset(train_df, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = LegalDataset(val_df, tokenizer, MAX_SEQ_LENGTH)
    test_dataset = LegalDataset(test_df, tokenizer, MAX_SEQ_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = RoleAwareTransformer(PRETRAINED_MODEL_NAME, NUM_ROLES)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Initialize EarlyStopping and TensorBoard writer
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, save_path=MODEL_SAVE_PATH)
    writer = SummaryWriter(log_dir='runs/RoleAwareTransformer')

    # Training loop with early stopping and TensorBoard logging
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch, writer)
        val_metrics = evaluate_model(model, val_dataloader, epoch, writer, phase='Validation')

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Metrics: Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
              f"F1 Score: {val_metrics['f1_score']:.4f}, MCC: {val_metrics['mcc']:.4f}")

        # Check early stopping condition
        early_stopping(val_metrics['loss'], model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Load the best model
    model.load_state_dict(torch.load('../../saved_models/Role-Aware/Role-Aware/'))

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_dataloader, phase='Test')
    print("\nTest Set Evaluation:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}")

    # Save the final model (if needed)
    torch.save(model.state_dict(), 'final_model.pth')

    # Close the TensorBoard writer
    writer.close()
