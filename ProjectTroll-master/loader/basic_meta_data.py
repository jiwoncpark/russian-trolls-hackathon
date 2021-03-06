import torch
import numpy as np
import os
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, Iterator, Pipeline
import sys
import csv
csv.field_size_limit(sys.maxsize)

# Deal with local/remote
if 'Users' in os.getcwd():
    # Specify your local mydata folder
    glove_path = '/Users/zhangyue/Desktop/russian-trolls-hackathon/ProjectTroll-master/data/mydata'
else:
    # Specify your cloud mydata folder
    glove_path = '/home/zyflame104/GloVe'
    
troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
sys.path.insert(0, troll_root)
glove_path = os.path.join(troll_root, '.vector_cache')

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
             ('right', LABEL),
             ('7', None),
             ('8', None),
             ('9', None)]
    
    #train_csv = 'twitter_pollster_'+str(obj.days)+'_days_train_small.csv'
    #test_csv = 'twitter_pollster_'+str(obj.days)+'_days_test_small.csv'
    train_csv = 'train1.csv'
    test_csv = 'test1.csv'
    
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
                                                 cache=glove_path))
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

if __name__== "__main__":

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True,
                 tokenize=tokenize,
                 lower=True,
                 batch_first=True,
                 fix_length=None)

    VARIABLE = Field(sequential=False,
                  dtype=torch.float,
                  batch_first=True,
                  use_vocab=False,)
    
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
             ('right', LABEL),
             ('7', None),
             ('8', None),
             ('9', None)]
    
    train_csv = 'train1.csv'
    test_csv = 'test1.csv'

    train_dataset = TabularDataset(path='mydata/'+train_csv,
                                   format='csv',
                                   skip_header=True,
                                   fields=fields)

    test_dataset = TabularDataset(path='mydata/'+test_csv,
                                  format='csv',
                                  skip_header=True,
                                  fields=fields)

    TEXT.build_vocab(train_dataset, vectors=GloVe(name='twitter.27B',
                                                  dim=25))
    vocab_size = len(TEXT.vocab)
    word_embeddings = TEXT.vocab.vectors
    print ("vector size of text vocabulary: ", TEXT.vocab.vectors.size())

    train_iter, test_iter = Iterator.splits(
            (train_dataset, test_dataset),
            sort_key=lambda x: len(x.content), 
            batch_sizes=(7, 7),
            device=torch.device('cpu'),
            sort_within_batch=True,
            repeat=False)
    
    print(train_csv, test_csv)
    train_iter_ = BatchWrapper(train_iter, ['content', 'avg_followers', 'avg_following', 'avg_left', 'avg_news', 'avg_right', 'time', 'baseline_pred_left', 'baseline_pred_mid', 'baseline_pred_right'], ['left', 'mid', 'right'])
    test_iter_ = BatchWrapper(test_iter, ['content', 'avg_followers', 'avg_following', 'avg_left', 'avg_news', 'avg_right', 'time', 'baseline_pred_left', 'baseline_pred_mid', 'baseline_pred_right'], ['left', 'mid', 'right'])

    batch0 = None
    batch1 = None
    batch2 = None
    for iter, batch in enumerate(train_iter_, 1):
        if iter==1:
            #print(iter, batch)
            batch0 = batch[0]
            batch1 = batch[1]
            batch2 = batch[2]
            print("batch[0]: ", batch[0])
            print("batch[1]: ", batch[1]) # 6 metadata
            print("batch[2]: ", batch[2])
            print("batch[0] size: ", batch[0].shape)
            print("batch[1] size: ", batch[1].shape)
            print("batch[2] size: ", batch[2].shape)
        break
    
    np.save('batch0', batch0)
    np.save('batch1', batch1)
    np.save('batch2', batch2)