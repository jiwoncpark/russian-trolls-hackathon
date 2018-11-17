import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, obj):
        super(LSTM, self).__init__()

        self.obj = obj
        
        self.word_embeddings = nn.Embedding(obj.vocab_size,
                                            obj.embedding_dim)
        
        self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(obj.embedding_dim, obj.hidden_size)
        
        self.label = nn.Linear(obj.hidden_size, obj.output_size)

    def forward(self, input):

        batch_size = input.shape[0]
        
        input = self.word_embeddings(input)
        input = input.permute(1, 0, 2)
        
        h_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))
        c_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))
        
        h_0 = h_0.to(self.obj.device)
        c_0 = c_0.to(self.obj.device)

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        
        return self.label(final_hidden_state[-1])

