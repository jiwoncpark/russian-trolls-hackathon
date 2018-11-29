import torch
import os
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, Iterator, Pipeline
import sys
import csv
csv.field_size_limit(sys.maxsize)

# Deal with local/remote
if 'Users' in os.getcwd():
    # Specify your local mydata folder
    path = '/Users/zhangyue/Desktop/russian-trolls-hackathon/ProjectTroll-master/data/mydata'
else:
    # Specify your cloud mydata folder
    path = '/home/zyflame104/GloVe'

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
        self.text_col = self.x_var[0]
        self.meta_cols = self.x_var[1:]
    
    def __iter__(self):
        for batch in self.dl:
            y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            x_text = getattr(batch, self.text_col)
            x_meta = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.meta_cols], dim=1).float()
            yield (x_text, x_meta, y)
  
    def __len__(self):
        return len(self.dl)
    
def basic_meta_data(obj):
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True,
                 tokenize=tokenize,
                 lower=True,
                 batch_first=True,
                 fix_length=obj.fix_length)

    VARIABLE = Field(sequential=False,
                  dtype=torch.float,
                  batch_first=True,
                  use_vocab=False)
    
    LABEL = Field(sequential=False,
                  dtype=torch.float,
                  batch_first=True,
                  use_vocab=False)
    
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
             ('right', LABEL),]
    
    #train_csv = 'twitter_pollster_'+str(obj.days)+'_days_train_small.csv'
    #test_csv = 'twitter_pollster_'+str(obj.days)+'_days_test_small.csv'
    train_csv = 'train'+str(obj.data_num)+'.csv'
    test_csv = 'test'+str(obj.data_num)+'.csv'
    
    train_dataset = TabularDataset(path=obj.data_path+'/'+train_csv,
                                   format='csv',
                                   skip_header=True,
                                   fields=fields)
    
    test_dataset = TabularDataset(path=obj.data_path+'/'+test_csv,
                                  format='csv',
                                  skip_header=True,
                                  fields=fields)
    
    TEXT.build_vocab(train_dataset, vectors=GloVe(name=obj.Glove_name,
                                                  dim=obj.embedding_dim, 
                                                 cache=path))
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
    
    train_iter_ = BatchWrapper(train_iter, ['content', 'avg_followers', 'avg_following', 'avg_left', 'avg_news', 'avg_right', 'time', 'baseline_pred_left', 'baseline_pred_mid', 'baseline_pred_right'], ['left', 'mid', 'right'])
    test_iter_ = BatchWrapper(test_iter, ['content', 'avg_followers', 'avg_following', 'avg_left', 'avg_news', 'avg_right', 'time', 'baseline_pred_left', 'baseline_pred_mid', 'baseline_pred_right'], ['left', 'mid', 'right'])
    
    return TEXT, vocab_size, word_embeddings, train_iter_, test_iter_


