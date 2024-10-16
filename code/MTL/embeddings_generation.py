## Importing all required libraries
import torch
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
import json
import nltk
from nltk import sent_tokenize
from tqdm import tqdm
nltk.download('punkt')

## Loading the model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

## Setting model to evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

## Loading data
## Change the paths s.t it matches your file path
train_path = 'train.json'
test_path = 'test.json'

## Loading the json data
with open(train_path, 'r') as f:
    train = json.load(f)
    f.close()
    
with open(test_path, 'r') as f:
    test = json.load(f)
    f.close()

train_files = list(train.keys())
test_files = list(test.keys())

out = 'bert_emb/' ## Give the path where you want to save the embeddings

# !mkdir 'train_bert_emb/'

## Getting the embeddings for case in train_it, similarly follow for other files
## Here we use the embedding corresponding to the [CLS] token as the sentences representation
for case in tqdm(train.keys()):
    sentences = train[case]['sentences']
    all_text = ""
    start_time = time.time()
    for idx in range(len(sentences)):
        
        text = sentences[idx]
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        if(len(tokenized_text) > 510):
            tokenized_text = tokenized_text[:510] + ['[SEP]']
            
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
        
        emb = outputs[0].squeeze()[0].flatten().tolist()
        
        emb = [str(round(i,5)) for i in emb]
        final = " ".join(emb)
        final += "\t"+train[case]['labels'][idx]
        all_text += final+"\n"
    with open(os.path.join(out, case),"w") as f:
        f.write(all_text)