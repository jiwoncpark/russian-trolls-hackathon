import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, obj):
        super(LSTM, self).__init__()

        self.obj = obj
        
        self.relu = nn.ReLU()
        
        self.bn_meta = nn.BatchNorm1d(obj.meta_dim)
        
        self.bn_concat = nn.BatchNorm1d(obj.hidden_size + obj.meta_hidden_size)
        
        self.bn_fc = nn.BatchNorm1d(obj.total_hidden_size)
        
        self.word_embeddings = nn.Embedding(obj.vocab_size,
                                            obj.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(obj.word_embeddings, requires_grad=False)
        #self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(obj.embedding_dim, obj.hidden_size)
        
        self.metafc = nn.Linear(obj.meta_dim, obj.meta_hidden_size)

        self.fc_1 = nn.Linear(obj.hidden_size + obj.meta_hidden_size, obj.total_hidden_size)

        self.fc_final = nn.Linear(obj.total_hidden_size, obj.output_size)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_text, input_meta):

        batch_size = input_text.shape[0]
        #meta_size = meta_text.shape[1]

        input_text = self.word_embeddings(input_text)
        input_text = input_text.permute(1, 0, 2)
        
        h_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))
        c_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))
        
        h_0 = h_0.to(self.obj.device)
        c_0 = c_0.to(self.obj.device)

        output, (final_hidden_state, final_cell_state) = self.lstm(input_text, (h_0, c_0))
        
        bn_input_meta = self.bn_meta(input_meta)
        output_meta = self.metafc(bn_input_meta)
        output_meta = self.relu(output_meta)

        if self.obj.last_sigmoid:
            concat = torch.cat((final_hidden_state[-1], output_meta), dim=1)
            bn_concat = self.bn_concat(concat)
            bn_concat = self.relu(bn_concat)
            after_fc_1 = self.fc_1(bn_concat)
            bn_after_fc_1 = self.bn_fc(after_fc_1)
            bn_after_fc_1 = self.relu(bn_after_fc_1)
            after_fc_final = self.fc_final(bn_after_fc_1)
            return after_fc_final #self.sigmoid(after_fc_final)
        else:
            return self.label(final_hidden_state[-1])

