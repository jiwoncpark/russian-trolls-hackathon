import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import sys
import csv
csv.field_size_limit(sys.maxsize)

class Encoder(nn.Module):
    """ Encoder containing the embedding plus LSTM layers.
    Note
    ----
    The classifying linear layer has been separated out of the
    LSTM class so this class is now called Encoder.
    
    """
    def __init__(self, obj):
        super(Encoder, self).__init__()

        self.obj = obj
        
        self.word_embeddings = nn.Embedding(obj.vocab_size,
                                            obj.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(obj.word_embeddings, requires_grad=False)
        #self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(obj.embedding_dim, obj.hidden_size)
        
    def forward(self, input):

        batch_size = input.shape[0]
        
        input = self.word_embeddings(input)
        input = input.permute(1, 0, 2)
        
        h_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))
        c_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))
        
        h_0 = h_0.to(self.obj.device)
        c_0 = c_0.to(self.obj.device)

        return self.lstm(input, (h_0, c_0))

class Attention(nn.Module):
    def __init__(self, obj):
        
        super(Attention, self).__init__()
        self.scale = 1.0 / math.sqrt(obj.attention_size)
        
    def forward(self, lstm_output, final_state):
        # Query ~ [B, Q]
        # Keys ~ [T, B, K]
        # Values ~ [T, B, V]
        # Outputs: a ~ [T, B], lin_comb ~[B, V]
        
        # Use dot product attention, so q_dim = k_dim (Q = K)
        
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #query = query.unsqueeze(1) # [B, Q] --> [B, 1, Q]
        #keys = keys.transpose(0, 1).transpose(1, 2) # [T, B, K] --> [B, K, T] inverse permuation op
        #energy = torch.bmm(query, keys) # [B, 1, Q] bmm [B, K=Q, T] --> [B, 1, T] batch matrix multiplication
        #energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        
        #values = values.transpose(0, 1) # [T, B, V] --> [B, T, V]
        #linear_combination = torch.bmm(energy, values).squeeze(1) # [B, 1, T] bmm [B, T, V] --> [B, 1, V] --> [B, V]
        print(soft_attn_weights.shape)
        return new_hidden_state

class AttentionLSTM(nn.Module):
    """Class combining LSTM + Attention + Classifier."""
    
    def __init__(self, obj):
        super(AttentionLSTM, self).__init__()
        self.encoder = Encoder(obj)
        self.attention = Attention(obj)
        self.metafc = nn.Linear(obj.meta_dim, obj.meta_hidden_size)
        self.fc_1 = nn.Linear(obj.hidden_size + obj.meta_hidden_size, obj.total_hidden_size)
        self.fc_final = nn.Linear(obj.total_hidden_size, obj.output_size)
        self.sigmoid = nn.Sigmoid()
        
        self.relu = nn.ReLU()
        self.bn_meta = nn.BatchNorm1d(obj.meta_dim)
        self.bn_concat = nn.BatchNorm1d(obj.hidden_size + obj.meta_hidden_size)
        self.bn_fc = nn.BatchNorm1d(obj.total_hidden_size)
    
    def forward(self, input_text, input_meta):
        #output, (final_hidden, state, final_cell_state) = self.encoder(input)
        output, final_state = self.encoder(input_text)
        output = output.permute(1, 0, 2)
        #hidden = final_cell_state[-1] # last layer of cell state
        # final hidden state
        #print(final_state[0].shape)
        bn_input_meta = self.bn_meta(input_meta)
        output_meta = self.metafc(bn_input_meta)
        output_meta = self.relu(output_meta)

        attn_output = self.attention(output, final_state[0])
        concat = torch.cat((attn_output, output_meta), dim=1)
        bn_concat = self.bn_concat(concat)
        bn_concat = self.relu(bn_concat)
        after_fc_1 = self.fc_1(bn_concat)
        
        bn_after_fc_1 = self.bn_fc(after_fc_1)
        bn_after_fc_1 = self.relu(bn_after_fc_1)
        after_fc_final = self.fc_final(bn_after_fc_1)

        return self.sigmoid(after_fc_final)

if __name__=='__main__':
    from torchtext.data import Field, TabularDataset, Iterator, Pipeline
    import os, sys
    import numpy as np
    from torchtext.vocab import GloVe
    
    class Options():
        def __init__(self):
            
            
            self.vocab_size = 20427
            self.embedding_dim = 25
            #self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.hidden_size = 64
            self.meta_hidden_size = 16
            self.meta_dim = 9
            self.total_hidden_size = 8
            self.device = 'cpu' #'cuda:0'
            self.output_size = 3
            self.Glove_name = 'twitter.27B'
            self.fix_length = None
            self.attention_size = 64
            
            troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
            sys.path.insert(0, troll_root)
            glove_path = os.path.join(troll_root, '.vector_cache')
            self.data_path = os.path.join(troll_root, 'mydata')    
            
            tokenize = lambda x: x.split()
            TEXT = Field(sequential=True,
                         tokenize=tokenize,
                         lower=True,
                         batch_first=True,
                         fix_length=self.fix_length)
            
            VARIABLE = Field(sequential=False,
                  dtype=torch.float,
                  batch_first=True,
                  use_vocab=False)

            LABEL = Field(sequential=False,
                          dtype=torch.float,
                          batch_first=True,
                          use_vocab=False)
            train_csv = 'train1.csv'
            #test_csv = 'test1.csv'
            
            fields = [#('id', None),
              ('content', TEXT),
              ('avg_followers',VARIABLE),
              ('avg_following', VARIABLE),
              ('avg_left', VARIABLE),
              ('avg_news', VARIABLE),
              ('avg_right', VARIABLE),
              ('time', VARIABLE),
              ('baseline_pred_left', VARIABLE),
              ('baseline_pred_mid', VARIABLE),
              ('baseline_pred_right', VARIABLE),
              ('left', LABEL),
             ('mid', LABEL),
             ('right', LABEL),
             ('7', None),
             ('8', None),
             ('9', None)]
            
            train_dataset = TabularDataset(path=self.data_path + '/' + train_csv,
                                           format='csv',
                                           skip_header=True,
                                           fields=fields)
            TEXT.build_vocab(train_dataset, vectors=GloVe(name=self.Glove_name,
                                                  dim=self.embedding_dim, 
                                                 cache=glove_path))
            #vocab_size = len(TEXT.vocab)
            self.word_embeddings = TEXT.vocab.vectors
    
    obj = Options()
    
    model = AttentionLSTM(obj)
    model.load_state_dict(torch.load('./results/net=AttentionLSTM-lr=0.01-total_loss.pth'))
    model.eval()
    input_meta_np = np.load('batch1.npy')
    input_meta = torch.from_numpy(input_meta_np)
    input_text_np = np.load('batch0.npy')
    #print(input_text_np.shape)
    input_text = torch.from_numpy(input_text_np) #torch.FloatTensor(input_text_np)
    
    est = model(input_text=input_text, input_meta=input_meta)
    print("est:", est)