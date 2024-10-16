## Importing required libraries
import json
from tqdm import tqdm
import pandas as pd
## some more imports
import progressbar
from keras.preprocessing.sequence import pad_sequences

f_train = open("train.json", "r")
f_test = open("test.json", "r")

## loading the data
data_tr = json.load(f_train)
f_train.close()
data_te = json.load(f_test)
f_test.close()

#### Data conversion #######

def json_to_df(data):
  sentences_1 = []
  sentences_2 = []
  label = []
  for doc in data.keys():
    length_sentences = len(data[doc]["sentences"])
    for i,sentence in enumerate(data[doc]["sentences"]):
      if(i== length_sentences-1):
        break
      sentences_1.append(data[doc]["sentences"][i])
      sentences_2.append(data[doc]["sentences"][i+1])
      label_1 = data[doc]["labels"][i]
      label_2 = data[doc]["labels"][i+1]
      if label_1 != label_2:
        label.append(1)
      else:
        label.append(0)

  df = pd.DataFrame(list(zip(sentences_1, sentences_2, label)), columns =['Sentence 1', 'Sentence 2', "label"])
  return df

## Converting out data from json to dataframe

train_df = json_to_df(data_tr)
test_df = json_to_df(data_te)

## importing relevant functions from transformers library that will be used
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)}

model_type = 'bert' ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = 'bert-base-uncased'

## loading our tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

'''
    Function to get imput ids for each sentences using the tokenizer
'''
def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []
  token_type_ids = []

  for i in tqdm(range(len(dataf['Sentence 1']))):
    sen1 = dataf['Sentence 1'].iloc[i]
    sen1_t = tokenizer.tokenize(sen1)
    sen2 = dataf['Sentence 2'].iloc[i]
    sen2_t = tokenizer.tokenize(sen2)
    if(len(sen1_t) > 253):
      sen1_t = sen1_t[:253]
    if(len(sen2_t) > 253):
      sen2_t = sen2_t[:253]
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token

    sen_full = [CLS] + sen1_t + [SEP] + sen2_t + [SEP]
    tok_type_ids_0 = [0 for i in range(len(sen1_t)+2)]
    tok_type_ids_1 = [1 for i in range(512-len(sen1_t)-2)]
    tok_type_ids = tok_type_ids_0 + tok_type_ids_1
    token_type_ids.append(tok_type_ids)
    encoded_sent = tokenizer.convert_tokens_to_ids(sen_full)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=512, value=0, dtype="long", truncating="pre", padding="post")
  # token_type_ids = pad_sequences(token_type_ids, maxlen=512, value=1, dtype="long", truncating="pre", padding="post")

  return input_ids, lengths, token_type_ids

## Getting input ids for train and validation set
train_input_ids, train_lengths, train_token_type_ids = input_id_maker(train_df, tokenizer)
validation_input_ids, validation_lengths, validation_token_type_ids = input_id_maker(test_df, tokenizer)

'''
    This functions returns the attention mask for given input id
'''
def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks

## getting attention masks and labels for train and val sentences
train_attention_masks = att_masking(train_input_ids)
validation_attention_masks = att_masking(validation_input_ids)

train_labels = train_df['label'].to_numpy().astype('int')
validation_labels = test_df['label'].to_numpy().astype('int')

## Imports
import torch
from sklearn.model_selection import train_test_split
# from google.colab import drive
import textwrap
import progressbar
import keras
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json

train_inputs = train_input_ids
validation_inputs = validation_input_ids
train_masks = train_attention_masks
validation_masks = validation_attention_masks
train_tti = train_token_type_ids
validation_tti = validation_token_type_ids

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_tti = torch.tensor(train_tti)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
validation_tti = torch.tensor(validation_tti)

## loading pretrained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
model.to(device)

# max batch size should be 6 due to colab limits
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_tti, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_tti, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)

import numpy as np
lr = 2e-5
max_grad_norm = 1.0
epochs = 3
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 2212


np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
        
        with torch.no_grad():        
          outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
    
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("")
print("Training complete!")

prediction_data = validation_data
prediction_sampler = validation_sampler
prediction_dataloader = validation_dataloader

prediction_inputs = validation_inputs
prediction_masks = validation_masks
prediction_labels = validation_labels

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
model.eval()

predictions , true_labels = [], []

for (step, batch) in enumerate(prediction_dataloader):
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
  
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                      attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')

predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()

flat_accuracy(predictions,true_labels)
from sklearn.metrics import classification_report
print(classification_report(labels_flat, pred_flat))

validation_input_ids, validation_lengths, validation_token_type_ids = input_id_maker(test_df, tokenizer)
validation_attention_masks = att_masking(validation_input_ids)
validation_labels = test_df['label'].to_numpy().astype('int')

validation_inputs = torch.tensor(validation_input_ids)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_attention_masks)
validation_tti = torch.tensor(validation_token_type_ids)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_tti, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)

prediction_inputs = validation_inputs
prediction_masks = validation_masks
prediction_labels = validation_labels

prediction_data = validation_data
prediction_sampler = validation_sampler
prediction_dataloader = validation_dataloader

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
model.eval()

predictions , true_labels = [], []

for (step, batch) in enumerate(prediction_dataloader):
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
  
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                      attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')

predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()
flat_accuracy(predictions,true_labels)
print(classification_report(labels_flat, pred_flat))

## Saving trained model
import os

output_dir = "SiameseBERT_7labels_full/" # path to which fine tuned model is to be saved

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

## Loading the saved model
model = BertForSequenceClassification.from_pretrained('SiameseBERT_7labels_full/', output_hidden_states=True)
model.to(device)

'''
    This function returns the [CLS] embedding for a given input_id and attention mask
'''
def get_output_for_one_vec(input_id, att_mask):
  input_ids = torch.tensor(input_id)
  att_masks = torch.tensor(att_mask)
  input_ids = input_ids.unsqueeze(0)
  att_masks = att_masks.unsqueeze(0)
  model.eval()
  input_ids = input_ids.to(device)
  att_masks = att_masks.to(device)
  with torch.no_grad():
      output = model(input_ids=input_ids, token_type_ids=None, attention_mask=att_masks)

  vec = output["hidden_states"][12][0][0]
  vec = vec.detach().cpu().numpy()
  return vec

## Getting embeddings for train sentences
clsembs_train = []
for i, ii in enumerate(train_input_ids):
  clsembs_train.append(get_output_for_one_vec(ii, train_attention_masks[i]))
  
  ## Getting embeddings for test sentences
clsembs_test = []
for i, ii in enumerate(validation_input_ids):
  clsembs_test.append(get_output_for_one_vec(ii, validation_attention_masks[i]))
  
  i=0 ## Loading the train embeddings 
for key in data_tr.keys():
  limit = len(data_tr[key]["sentences"])
  sp = clsembs_train[i:i+limit-1]
  np.save("SiameseBERT_7labels_full/" + key[:-4] + "train", np.array(sp))
  i = i+limit-1
  
  i=0 ## ## Loading the train embeddings
for key in data_te.keys():
  limit = len(data_te[key]["sentences"])
  sp = clsembs_test[i:i+limit-1]
  np.save("SiameseBERT_7labels_full/" + key[:-4] + "test", np.array(sp))
  i = i+limit-1