import datetime
import pollster
import sys, os
troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
sys.path.insert(0, troll_root)

api = pollster.Api()

# choose the 2016 election
charts = api.charts_get(
  cursor=None,
  tags='2016-president',
  election_date=datetime.date(2016, 11, 8)
)

# choose a question with more than 600 polls
# this will give us the question of Trump vs Clinton
question_slug = next(c.question.slug for c in charts.items if c.question.n_polls > 600)

print()
print('GET /question/:slug/poll_responses_clean.tsv... (for %s)' % question_slug)

responses_clean = api.questions_slug_poll_responses_clean_tsv_get(question_slug)


# remove all columns except trump, clinton and the starting date of the poll
#responses_clean = responses_clean[['start_date','Trump','Clinton']]

responses_clean.to_csv(os.path.join(troll_root, 'mydata', 'pollster.csv'))

print('  Found %d responses to Question %s' % (len(responses_clean), question_slug))
print(repr(responses_clean[0:5]))
print('...')

