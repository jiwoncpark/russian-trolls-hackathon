import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta
import sys, os
troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
sys.path.insert(0, troll_root)

days = 7
election_date = pd.to_datetime('2016-11-08')

# load both datasets
print('loading data')
twitter = pd.read_csv(os.path.join(troll_root, 'mydata', 'twitter_metadata.csv'))
pollster = pd.read_csv(os.path.join(troll_root, 'mydata', 'pollster_left_mid_right.csv'))

# format dates
print('formating dates')
twitter['publish_date'] = pd.to_datetime(twitter['publish_date'])
pollster['start_date'] = pd.to_datetime(pollster['start_date'])

pollster.sort_values(by=['start_date'], inplace=True)

# assign tweets uniquely to polls
print('assigning tweets to polls')
start_time = time.time()
for i in range(len(twitter)):
    # find all polls that this tweet affects
    twitter_publish = twitter.loc[i,'publish_date']

    poll_idxs = pollster.loc[(twitter_publish < pollster['start_date']) &\
    (pollster['start_date'] < twitter_publish + timedelta(days=days))].index.values
             
    # assign tweet randomly to one of the polls
    if len(poll_idxs) > 0:
        twitter.loc[i,'poll_idx'] = np.random.choice(poll_idxs)
    else:
        twitter.loc[i,'poll_idx'] = None

twitter = twitter.loc[twitter['poll_idx'] != None]

# twitter = twitter.join(pollster,on='poll_idx')
# twitter.sort_values(by=['poll_idx'], inplace=True)
# twitter.to_csv(os.path.join(troll_root, 'mydata', 'twitter_pollster.csv'))
 
# aggregate the tweets of each poll
print('aggregating tweets belonging to the same poll')
polls_with_tweets = []

for j in range(len(pollster)):
    tweets = twitter[twitter['poll_idx'] == j]
    print(len(tweets))
    
    if len(tweets) > 0:
        polls_with_tweets.append(j)
        pollster.loc[j,'content'] = ' '.join(tweets['content'].values)
        pollster.loc[j,'avg_followers'] = np.mean(tweets['followers'].values)
        pollster.loc[j,'avg_following'] = np.mean(tweets['following'].values)
        pollster.loc[j,'avg_right'] = np.mean(tweets['account_category'] == 'RightTroll')
        pollster.loc[j,'avg_left'] = np.mean(tweets['account_category'] == 'LeftTroll')
        pollster.loc[j,'avg_news'] = np.mean(tweets['account_category'] == 'NewsFeed')
        pollster.loc[j,'time'] = (election_date - pollster.loc[j,'start_date']).days

print(pollster.columns)
        
for col in ['avg_followers', 'avg_following', 'avg_right', 'avg_left', 'avg_news', 'time', 'left', 'mid', 'right']:
    pollster[col] = pollster[col].astype(np.float32)
        
# remove polls with no tweets
pollster = pollster.loc[polls_with_tweets]

# remove unnecessary columns
pollster = pollster[['content','avg_followers','avg_following',\
            'avg_right','avg_left','avg_news','time','left','mid','right']]

# split to train and test polls
idx = int(0.8*len(pollster))

train = pollster.iloc[:idx]
test = pollster.iloc[idx:]

# save to csv
train.to_csv(os.path.join(troll_root, 'mydata', 'twitter_pollster_' + str(days) + '_days_train.csv'))
test.to_csv(os.path.join(troll_root, 'mydata', 'twitter_pollster_' + str(days) + '_days_test.csv'))

