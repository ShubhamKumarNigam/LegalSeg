
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import time
import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import string
from collections import defaultdict
import os
import warnings

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

'''
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. The feature vectors are actually context-aware sentence embeddings. These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter_Binary(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.hidden = None
        self.device = device
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sequences):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        
        # generate context-aware sentence embeddings (feature vectors)
        ## tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        x_new = self.dropout(x)
        
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        x_new = self.hidden2tag(x_new)
        return x_new, x


'''
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. The feature vectors are actually context-aware sentence embeddings. These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(2*hidden_dim, n_tags)
        self.hidden = None
        self.device = device
        self.emb_dim = emb_dim
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sequences, hidden_binary):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        
        # generate context-aware sentence embeddings (feature vectors)
        ## tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        final = torch.zeros((x.shape[0], x.shape[1], 2*x.shape[2])).to(self.device)
        for batch_name, doc in enumerate(x):
            for i, sent in enumerate(doc):
                final[batch_name][i] = torch.cat((x[batch_name][i], hidden_binary[batch_name][i]),0)
        final = self.dropout(final)
        
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        final = self.hidden2tag(final)
        return final

'''
    A linear-chain CRF is fed with the emission scores at each sentence, and it finds out the optimal sequence of tags by learning the transition scores.
'''
class CRF(nn.Module):    
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx = None):
        super().__init__()
        
        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx
        
        self.transitions = nn.Parameter(torch.empty(self.n_tags, self.n_tags))
        self.init_weights()
        
    def init_weights(self):
        # initialize transitions from random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
        # enforce constraints (rows = from, cols = to) with a big negative number.
        # exp(-1000000) ~ 0
        
        # no transitions to SOS
        self.transitions.data[:, self.SOS_TAG_IDX] = -1000000.0
        # no transition from EOS
        self.transitions.data[self.EOS_TAG_IDX, :] = -1000000.0
        
        if self.PAD_TAG_IDX is not None:
            # no transitions from pad except to pad
            self.transitions.data[self.PAD_TAG_IDX, :] = -1000000.0
            self.transitions.data[:, self.PAD_TAG_IDX] = -1000000.0
            # transitions allowed from end and pad to pad
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0
            
    def forward(self, emissions, tags, mask = None):
        ## emissions: tensor[batch_size, seq_len, n_tags]
        ## tags: tensor[batch_size, seq_len]
        ## mask: tensor[batch_size, seq_len], indicates valid positions (0 for pad)
        return -self.log_likelihood(emissions, tags, mask = mask)
    
    def log_likelihood(self, emissions, tags, mask = None):                   
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
            
        scores = self._compute_scores(emissions, tags, mask = mask)
        partition = self._compute_log_partition(emissions, mask = mask)
        return torch.sum(scores - partition)
    
    # find out the optimal tag sequence using Viterbi Decoding Algorithm
    def decode(self, emissions, mask = None):      
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
            
        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences
    
    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        if(torch.cuda.is_available()):
            scores = torch.zeros(batch_size).cuda()
        else:
            scores = torch.zeros(batch_size)
        
        # save first and last tags for later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        
        # add transition from SOS to first tags for each sample in batch
        t_scores = self.transitions[self.SOS_TAG_IDX, first_tags]
        
        # add emission scores of the first tag for each sample in batch
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores
        
        # repeat for every remaining word
        for i in range(1, seq_len):
            
            is_valid = mask[:, i]
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            
            e_scores = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[prev_tags, curr_tags]
                        
            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            
            scores += e_scores + t_scores
            
        # add transition from last tag to EOS for each sample in batch
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]
        return scores
    
    # compute the partition function in log-space using forward algorithm
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        
        # in the first step, SOS has all the scores
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        
        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1) 
            
            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            
            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim = 1)
            
            # set alphas if the mask is valid, else keep current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
            
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)
        
        # return log_sum_exp
        return torch.logsumexp(end_scores, dim = 1)
    
    # return a list of optimal tag sequence for each example in the batch
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        
        # in the first iteration, SOS will have all the scores and then, the max
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        
        backpointers = []
        
        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1) 
            
            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            
            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            
            scores = e_scores + t_scores + a_scores
            
            # find the highest score and tag, instead of log_sum_exp
            max_scores, max_score_tags = torch.max(scores, dim = 1)
            
            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            
            backpointers.append(max_score_tags.t())
            
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences
    
    # auxiliary function to find the best path sequence for a specific example
    def _find_best_path(self, sample_id, best_tag, backpointers):
        ## backpointers: list[tensor[seq_len_i - 1, n_tags, batch_size]], seq_len_i is the length of the i-th sample of the batch
        
        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path


'''
    Top-level module which uses a Hierarchical-LSTM-CRF to classify.
    
    If pretrained = False, each example is represented as a sequence of sentences, which themselves are sequences of word tokens. Individual sentences are passed to LSTM_Sentence_Encoder to generate sentence embeddings. 
    If pretrained = True, each example is represented as a sequence of fixed-length pre-trained sentence embeddings.
    
    Sentence embeddings are then passed to LSTM_Emitter to generate emission scores, and finally CRF is used to obtain optimal tag sequence. 
    Emission scores are fed to the CRF to generate optimal tag sequence.
'''
class Hier_LSTM_CRF_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size = 0, pad_word_idx = 0, pretrained = False, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx
        
        # sentence encoder is not required for pretrained embeddings
#         self.sent_encoder = LSTM_Sentence_Encoder(vocab_size, word_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device) if not self.pretrained else None
            
        self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)
        
        self.emitter_binary = LSTM_Emitter_Binary(5, 3*sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf_binary = CRF(5, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)
        
    
    def forward(self, x, x_binary):
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)
        
            
        ## x: list[batch_size, sents_per_doc, sent_emb_dim]
        tensor_x = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x]
        tensor_x_binary = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x_binary]
        
        ## list[batch_size, sents_per_doc, sent_emb_dim] --> tensor[batch_size, max_seq_len, sent_emb_dim]
        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first = True).to(self.device)    
        tensor_x_binary = nn.utils.rnn.pad_sequence(tensor_x_binary, batch_first = True).to(self.device)  
        
        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1
        
        self.emissions_binary, self.hidden_binary = self.emitter_binary(tensor_x_binary)
        self.emissions = self.emitter(tensor_x, self.hidden_binary)
        
        _, path = self.crf.decode(self.emissions, mask = self.mask)
        _, path_binary = self.crf_binary.decode(self.emissions_binary, mask = self.mask)
        return path, path_binary
    
    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype = torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)
        
        nll = self.crf(self.emissions, tensor_y, mask = self.mask)
        return nll    
    
    def _loss_binary(self, y_binary):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y_binary = [torch.tensor(doc, dtype = torch.long) for doc in y_binary]
        tensor_y_binary = nn.utils.rnn.pad_sequence(tensor_y_binary, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)
        
        nll_binary = self.crf_binary(self.emissions_binary, tensor_y_binary, mask = self.mask)
        return nll_binary    

'''
    This file prepares the numericalized data in the form of lists, to be used in training mode.
    idx_order is the order of documents in the dataset.
        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
        y:  list[num_docs, sentences_per_doc]
'''
def prepare_data_new(idx_order_train, idx_order_val, idx_order_test, args, path_train, path_val, path_test, dim, tag2idx=None):
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []
    
    
    word2idx = defaultdict(lambda: len(word2idx))
    if tag2idx is None:
        tag2idx = defaultdict(lambda: len(tag2idx))
        tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2
    

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'] = 0, 1

    # iterate over documents
    for doc in idx_order_train:
        doc_x, doc_y = [], [] 

        with open(path_train + doc) as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                	sent_x, sent_y = sent.strip().split('\t')
                except ValueError:
                	continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: word2idx[x], sent_x.split()))
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:dim]))
                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x_train.append(doc_x)
        y_train.append(doc_y)
   

    for doc in idx_order_val:
        doc_x, doc_y = [], [] 

        with open(path_val + doc) as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                	sent_x, sent_y = sent.strip().split('\t')
                except ValueError:
                	continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: word2idx[x], sent_x.split()))
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:dim]))
                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x_val.append(doc_x)
        y_val.append(doc_y)


    for doc in idx_order_test:
        doc_x, doc_y = [], [] 

        with open(path_test + doc) as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                	sent_x, sent_y = sent.strip().split('\t')
                except ValueError:
                	continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: word2idx[x], sent_x.split()))
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:dim]))
                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x_test.append(doc_x)
        y_test.append(doc_y)

    return x_train, y_train, x_val, y_val, x_test, y_test, word2idx, tag2idx


def batchify(x, y, x_binary, y_binary, batch_size):
    idx = list(range(len(x)))
    random.shuffle(idx)
    
    # convert to numpy array for ease of indexing
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    x_binary = np.array(x_binary)[idx]
    y_binary = np.array(y_binary)[idx]
    
    i = 0
    while i < len(x):
        j = min(i + batch_size, len(x))
        
        batch_idx = idx[i : j]
        batch_x = x[i : j]
        batch_y = y[i : j]
        
        batch_x_binary = x_binary[i : j]
        batch_y_binary = y_binary[i : j]
        
        yield batch_idx, batch_x, batch_y, batch_x_binary, batch_y_binary
        
        i = j


'''
    Perform a single training step by iterating over the entire training data once. Data is divided into batches.
'''
def train_step(model, opt, x, y, x_binary, y_binary, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.train()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    y_pred_binary = []
    y_gold_binary = []
    idx = [] # example index
    mu = 0.3
    
    for i, (batch_idx, batch_x, batch_y, batch_x_binary, batch_y_binary) in enumerate(batchify(x, y, x_binary, y_binary, batch_size)):
        pred, pred_binary = model(batch_x, batch_x_binary)
        loss = model._loss(batch_y)  
        loss_binary = model._loss_binary(batch_y_binary)
        
        overall = torch.add(torch.mul(loss, (1-mu)), torch.mul(loss_binary, mu))

        opt.zero_grad()
        overall.backward()
        opt.step()
        
        total_loss += overall.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        y_pred_binary.extend(pred_binary)
        y_gold_binary.extend(batch_y_binary)
        idx.extend(batch_idx)
    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    
    return total_loss / (i + 1), idx, y_gold, y_pred, y_gold_binary, y_pred_binary

'''
    Perform a single evaluation step by iterating over the entire training data once. Data is divided into batches.
'''
def val_step(model, x, y, x_binary, y_binary, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.eval()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    y_pred_binary = []
    y_gold_binary = []
    idx = [] # example index
    mu = 0.3
    
    for i, (batch_idx, batch_x, batch_y, batch_x_binary, batch_y_binary) in enumerate(batchify(x, y, x_binary, y_binary, batch_size)):
        pred, pred_binary = model(batch_x, batch_x_binary)
        loss = model._loss(batch_y)  
        loss_binary = model._loss_binary(batch_y_binary)
        
        overall = torch.add(torch.mul(loss, (1-mu)), torch.mul(loss_binary, mu))
        
        total_loss += overall.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        y_pred_binary.extend(pred_binary)
        y_gold_binary.extend(batch_y_binary)
        idx.extend(batch_idx)
        
    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    
    return total_loss / (i + 1), idx, y_gold, y_pred, y_gold_binary, y_pred_binary

'''
    Infer predictions for un-annotated data
'''
def infer_step(model, x):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    
    model.eval()
    y_pred = model(x) # predictions
    
    return y_pred    


'''
    Report all metrics in format using sklearn.metrics.classification_report
'''
def statistics(data_state, tag2idx):
    idx, gold, pred = data_state['idx'], data_state['gold'], data_state['pred']
    
    rev_tag2idx = {v: k for k, v in tag2idx.items()}
    tags = [rev_tag2idx[i] for i in range(len(tag2idx)) if rev_tag2idx[i] not in ['<start>', '<end>', '<pad>']]
    
    # flatten out
    gold = sum(gold, [])
    pred = sum(pred, [])
    
    
    print(classification_report(gold, pred, target_names = tags, digits = 3))

'''
    Train the model on entire dataset and report loss and macro-F1 after each epoch.
'''
def learn(model, train_x, train_y, val_x, val_y, test_x, test_y, tag2idx, train_x_binary, train_y_binary, val_x_binary, val_y_binary, test_x_binary, test_y_binary, tag2idx_binary, args):
    
     opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)
    
     print("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1'))
     print("-----------------------------------------------------------")
    
     best_val_f1 = 0.0
    
     model_state = {}
     data_state = {}
    
     start_time = time.time()
    
     for epoch in range(1, args.epochs + 1):

         train_loss, train_idx, train_gold, train_pred, train_gold_binary, train_pred_binary = train_step(model, opt, train_x, train_y, train_x_binary, train_y_binary, args.batch_size)
         val_loss, val_idx, val_gold, val_pred, val_gold_binary, val_pred_binary = val_step(model, val_x, val_y, val_x_binary, val_y_binary, args.batch_size)

         train_f1 = f1_score(sum(train_gold, []), sum(train_pred, []), average = 'macro')
         val_f1 = f1_score(sum(val_gold, []), sum(val_pred, []), average = 'macro')

         if epoch % args.print_every == 0:
             print("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f}".format(epoch, train_loss, train_f1, val_loss, val_f1))

         if val_f1 > best_val_f1:
             best_val_f1 = val_f1
             model_state = {'epoch': epoch, 'arch': model, 'name': model.__class__.__name__, 'state_dict': model.state_dict(), 'best_f1': val_f1, 'optimizer' : opt.state_dict()}
             data_state = {'idx': val_idx, 'loss': val_loss, 'gold': val_gold, 'pred': val_pred}
            
     end_time = time.time()
    
     print("Dumping model and data ...", end = ' ')
    
     torch.save(model_state, args.save_path + 'model_state' + '.tar')
    
     with open(args.save_path + 'data_state' + '.json', 'w') as fp:
         json.dump(data_state, fp)
    
     print("Done")    

     print('Time taken:', int(end_time - start_time), 'secs')
    
     statistics(data_state, tag2idx)
    
    model_best = Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size = 2, pretrained = args.pretrained, device = args.device).to(args.device)

    model_state = torch.load(args.save_path + 'model_state.tar')

    model_best.load_state_dict(model_state['state_dict'])

    test_loss, test_idx, test_gold, test_pred, test_gold_binary, test_pred_binary = val_step(model_best, test_x, test_y, test_x_binary, test_y_binary, args.batch_size)
    
    data_state = {'idx': test_idx, 'loss': test_loss, 'gold': test_gold, 'pred': test_pred}
    
    statistics(data_state, tag2idx)


class Args:
    pretrained = True
    data_path = '../../data/MTL/data/mtl-data/' ## Input to the pre=trained embedding(should contain 4 sub-folders, IT test and train, CL test and train)
    save_path = '../../saved_models/MTL/' ## path to save the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    print_every = 1
    lr = 0.005
    reg = 0
    emb_dim = 768
    epochs = 3

import warnings
import numpy as np

args = Args()
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# !mkdir '/kaggle/working/saved/'



## This cell just given the training and test files.
import json

train_path = '../../data/MTL/data/json/train.json'
test_path = '../../data/MTL/data/json/test.json'
val_path = '../../data/MTL/data/json/val.json'

train = {}
test = {}
val = {}

with open(train_path, 'r') as f:
    train = json.load(f)
    f.close()
    
with open(test_path, 'r') as f:
    test = json.load(f)
    f.close()
    
with open(val_path, 'r') as f:
    val = json.load(f)
    f.close()


train_files = list(train.keys())
test_files = list(test.keys())
val_files = list(val.keys())


print('\nPreparing data ...', end = ' ')

x_train, y_train, x_val, y_val, x_test, y_test, word2idx, tag2idx = prepare_data_new(train_files, val_files, test_files, args, os.path.join(args.data_path, 'train_emb/'), os.path.join(args.data_path, 'val_emb/'), os.path.join(args.data_path, 'test_emb/'), args.emb_dim)

x_train_binary, y_train_binary, x_val_binary, y_val_binary, x_test_binary, y_test_binary, word2idx_binary, tag2idx_binary = prepare_data_new(train_files, val_files, test_files, args, os.path.join(args.data_path,'train_binary/'), os.path.join(args.data_path,'val_binary/'), os.path.join(args.data_path,'test_binary/'), 3*args.emb_dim)

print('Done')

print('Vocabulary size Overall:', len(word2idx))
print('#Tags Overall:', len(tag2idx))

print('Vocabulary binary size Overall:', len(word2idx_binary))
print('#Tags Overall binary:', len(tag2idx_binary))

print('Dump word2idx and tag2idx')
with open(args.save_path + 'word2idx.json', 'w') as fp:
    json.dump(word2idx, fp)
with open(args.save_path + 'tag2idx.json', 'w') as fp:
    json.dump(tag2idx, fp)
    
with open(args.save_path + 'word2idx_binary.json', 'w') as fp:
    json.dump(word2idx_binary, fp)
with open(args.save_path + 'tag2idx_binary.json', 'w') as fp:
    json.dump(tag2idx_binary, fp)

print('\nInitializing model for Overall ...', end = ' ')   
model = Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size = len(word2idx), pretrained = args.pretrained, device = args.device).to(args.device)
print('Done')

print('\nEvaluating on test...')        
learn(model, x_train, y_train, x_val, y_val, x_test, y_test, tag2idx, x_train_binary, y_train_binary, x_val_binary, y_val_binary, x_test_binary, y_test_binary, tag2idx_binary, args)
