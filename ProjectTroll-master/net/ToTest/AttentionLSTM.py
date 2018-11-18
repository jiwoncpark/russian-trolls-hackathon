import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

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
        
        self.word_embeddings.weight.requires_grad = False
        
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
    
    def forward(self, input，meta):
        #output, (final_hidden, state, final_cell_state) = self.encoder(input)
        output, final_state = self.encoder(input)
        output = output.permute(1, 0, 2)
        #hidden = final_cell_state[-1] # last layer of cell state
        # final hidden state
        #print(final_state[0].shape)
        output_meta = self.metafc(meta)

        attn_output = self.attention(output, final_state[0])
        after_fc_1 = self.fc_1(torch.cat((attn_output, output_meta),0))
        after_fc_final = self.fc_final(after_fc_1)

        return self.sigmoid(after_fc_final)
    