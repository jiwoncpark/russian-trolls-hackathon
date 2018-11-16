import torch

from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, Iterator, Pipeline

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
    
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)
            y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            yield (x, y)
  
    def __len__(self):
        return len(self.dl)
    
def basic(obj):
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True,
                 tokenize=tokenize,
                 lower=True,
                 batch_first=True,
                 fix_length=obj.fix_length)
    
    LABEL = Field(sequential=False,
                  dtype=torch.float,
                  batch_first=True,
                  use_vocab=False)
    
    fields = [('id', None),
              ('content', TEXT),
              ('Trump', LABEL),
              ('Clinton', LABEL)]
    
    train_csv = 'twitter_pollster_'+str(obj.days)+'_days_train.csv'
    test_csv = 'twitter_pollster_'+str(obj.days)+'_days_test.csv'
    
    train_dataset = TabularDataset(path=obj.data_path+'/'+train_csv,
                                   format='csv',
                                   skip_header=True,
                                   fields=fields)
    
    test_dataset = TabularDataset(path=obj.data_path+'/'+test_csv,
                                  format='csv',
                                  skip_header=True,
                                  fields=fields)
    
    TEXT.build_vocab(train_dataset, vectors=GloVe(name=obj.Glove_name,
                                                  dim=obj.embedding_dim))
    vocab_size = len(TEXT.vocab)
    word_embeddings = TEXT.vocab.vectors
    print ("vector size of text vocabulary: ", TEXT.vocab.vectors.size())
    
    train_iter, test_iter = Iterator.splits(
            (train_dataset, test_dataset),
            sort_key=lambda x: len(x.content), 
            batch_sizes=(obj.train_batch_size, obj.test_batch_size),
            device=torch.device(obj.device),
            sort_within_batch=True,
            repeat=False)
    
    train_iter_ = BatchWrapper(train_iter, 'content', ['Trump','Clinton'])
    test_iter_ = BatchWrapper(test_iter, 'content', ['Trump','Clinton'])
    
    return TEXT, vocab_size, word_embeddings, train_iter_, test_iter_

