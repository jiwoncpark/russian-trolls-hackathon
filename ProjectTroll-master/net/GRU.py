import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU(nn.Module):
    def __init__(self, obj):
        super(GRU, self).__init__()

        self.obj = obj

        self.word_embeddings = nn.Embedding(obj.vocab_size, obj.embedding_dim)

        self.word_embeddings.weight.requires_grad = False

        self.gru = nn.GRU(obj.embedding_dim, obj.hidden_size)

        self.metafc = nn.Linear(obj.meta_dim, obj.meta_hidden_size)

        self.fc_1 = nn.Linear(obj.hidden_size + obj.meta_hidden_size, obj.total_hidden_size)

        self.fc_final = nn.Linear(obj.total_hidden_size, obj.output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_text, input_meta):

        batch_size = input_text.shape[0]
        #meta_size = meta.shape[0]

        input_text = self.word_embeddings(input_text)
        input_text = input_text.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, batch_size, self.obj.hidden_size))

        h_0 = h_0.to(self.obj.device)

        output, final_hidden_state = self.gru(input_text, h_0)

        output_meta = self.metafc(input_meta)

        after_fc_1 = self.fc_1(torch.cat((final_hidden_state[-1], output_meta), dim=1))

        after_fc_final = self.fc_final(after_fc_1)

        return self.sigmoid(after_fc_final)