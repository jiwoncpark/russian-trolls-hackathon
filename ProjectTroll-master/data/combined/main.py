import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta
import sys, os
troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
sys.path.insert(0, troll_root)

days = 7

# load both datasets
print('loading data')
twitter = pd.read_csv(os.path.join(troll_root, 'mydata', 'twitter.csv'))
pollster = pd.read_csv(os.path.join(troll_root, 'mydata', 'pollster.csv'))

# format dates
print('formating dates')
twitter['publish_date'] = pd.to_datetime(twitter['publish_date'])
pollster['start_date'] = pd.to_datetime(pollster['start_date'])
# for i in range(len(twitter)):
#     twitter.loc[i,'publish_date'] = datetime.strptime(
#             twitter.loc[i,'publish_date'],
#             '%m/%d/%Y %H:%M')

# for j in range(len(pollster)):
#     pollster.loc[j,'start_date'] = datetime.strptime(
#             pollster.loc[j,'start_date'],
#             '%Y-%m-%d')

pollster.sort_values(by=['start_date'], inplace=True)

# assign tweets uniquely to polls
print('assigning tweets to polls')
start_time = time.time()
for i in range(len(twitter)):
    # find all polls that this tweet affects
    twitter_publish = twitter.loc[i,'publish_date']

    poll_idxs = pollster.loc[(twitter_publish < pollster['start_date']) &\
    (pollster['start_date'] < twitter_publish + timedelta(days=days))].index.values

    # poll_idxs = []
    # for j in range(len(pollster)):
    #     pollster_start = pollster.loc[j,'start_date']
    #     if (twitter_publish < pollster_start) \
    #      & (twitter_publish > pollster_start - timedelta(days=days)):
    #          poll_idxs.append(j)
             
    # assign tweet randomly to one of the polls
    if len(poll_idxs) > 0:
        twitter.loc[i,'poll_idx'] = np.random.choice(poll_idxs)
    else:
        twitter.loc[i,'poll_idx'] = None
print(time.time()-start_time)  
# aggregate the tweets of each poll
print('aggregating tweets belonging to the same poll')
polls_with_tweets = []

for j in range(len(pollster)):
    tweets = twitter[twitter['poll_idx'] == j]
    
    if len(tweets) > 0:
        polls_with_tweets.append(j)
        
        pollster.loc[j,'content'] = ' '.join(tweets['content'].values)
        
# remove polls with no tweets
pollster = pollster.loc[polls_with_tweets]

# remove unnecessary columns
pollster = pollster[['content','Trump','Clinton']]

# split to train and test polls
idx = int(0.8*len(pollster))

train = pollster.iloc[:idx]
test = pollster.iloc[idx:]

# save to csv
train.to_csv(os.path.join(troll_root, 'mydata', 'twitter_pollster_' + str(days) + '_days_train.csv'))
test.to_csv(os.path.join(troll_root, 'mydata', 'twitter_pollster_' + str(days) + '_days_test.csv'))

