import sys

import pandas as pd
import numpy as np
from scipy import stats
from functools import partial


def load_tasks(tasks_filename):
    return pd.read_csv(tasks_filename, sep="\t")


import scipy.stats as st

def calc(row, hours24, history, users):
  ans = 0
  coeff = 0
  for h in range(row['hour_start'], row['hour_end'] + 1):
    coeff += hours24['count'].iloc[h%24]
  coeff /= (row['duration_hours'] * hours24['count'].mean())
  STATS_DURATION = history['hour'].max() - history['hour'].min()
  for user in row['user_ids']:
    cpms = np.array([])
    for pub_ in row['publishers']:
      if (pub_ == 0):
        continue
      cpms = np.concatenate((cpms, users[f'publisher_{int(pub_)}'].iloc[user]))
    mean = 0
    std = 0
    if len(cpms) == 0:
      continue
    elif len(cpms) == 1:
      mean = np.mean(cpms)
      std = 1e-7
    else:
      mean = np.mean(cpms)
      std = max(np.std(cpms), 1e-7)
    p_win = st.norm.cdf(np.log(row['cpm']), mean, std) * 0.51
    if np.isnan(p_win):
      print(cpms, mean, std)
    n_aucions = len(cpms) / STATS_DURATION * row['duration_hours'] * coeff * 1.1
    ans += (1 - (1 - p_win)**n_aucions)
  return ans / row['audience_size']

def calc2(row, hours24, history, users):
  ans = 0
  coeff = 0
  for h in range(row['hour_start'], row['hour_end'] + 1):
    coeff += hours24['count'].iloc[h%24]
  coeff /= (row['duration_hours'] * hours24['count'].mean())
  STATS_DURATION = history['hour'].max() - history['hour'].min()
  for user in row['user_ids']:
    cpms = np.array([])
    for pub_ in row['publishers']:
      if (pub_ == 0):
        continue
      cpms = np.concatenate((cpms, users[f'publisher_{int(pub_)}'].iloc[user]))
    if len(cpms) == 0:
      continue
    elif len(cpms) == 1:
      mean = np.mean(cpms)
      std = 1e-7
    else:
      mean = np.mean(cpms)
      std = max(np.std(cpms), 1e-7)
    p_win = min(1, st.norm.cdf(np.log(row['cpm']), mean, std) * 0.15)
    if np.isnan(p_win):
      print(cpms, mean, std)
    n_aucions = len(cpms) / STATS_DURATION * row['duration_hours'] * coeff
    ans += (1 - (1 - p_win)**n_aucions)
    # if n_aucions < 1:
    #   continue
    # print("*****", n_aucions)
    # ans += n_aucions * (p_win * (1 - p_win)**(n_aucions - 1))
  return ans / row['audience_size']

def calc3(row, hours24, history, users):
  ans = 0
  coeff = 0
  for h in range(row['hour_start'], row['hour_end'] + 1):
    coeff += hours24['count'].iloc[h%24]
  coeff /= (row['duration_hours'] * hours24['count'].mean())
  STATS_DURATION = history['hour'].max() - history['hour'].min()
  for user in row['user_ids']:
    cpms = np.array([])
    for pub_ in row['publishers']:
      if (pub_ == 0):
        continue
      cpms = np.concatenate((cpms, users[f'publisher_{int(pub_)}'].iloc[user]))
    mean = 0
    std = 0
    if len(cpms) == 0:
      continue
    elif len(cpms) == 1:
      mean = np.mean(cpms)
      std = 1e-7
    else:
      mean = np.mean(cpms)
      std = max(np.std(cpms), 1e-7)
    p_win = st.norm.cdf(np.log(row['cpm']), mean, std) * 0.08
    if np.isnan(p_win):
      print(cpms, mean, std)
    n_aucions = len(cpms) / STATS_DURATION * row['duration_hours'] * coeff * 0.08
    ans += (1 - (1 - p_win)**n_aucions)
  return ans / row['audience_size']

def main():
    tests_filename = sys.argv[1]

    val = load_tasks(tests_filename)

    # read data
    history = pd.read_csv('history.tsv', sep='\t')
    users = pd.read_csv('users.tsv', sep='\t')

    # preprocess history and user
    history['hour_mod_24'] = history['hour'] % 24
    history['log_cpm'] = np.log(history['cpm'])
    hours24 = history.groupby('hour_mod_24')['log_cpm'].agg(["mean", "count", "std"]).reset_index()

    user_publisher_stat = history.groupby(by=['user_id', 'publisher'])['log_cpm'].apply(list).reset_index()
    for pub_ in history.publisher.unique():
        users[f'publisher_{int(pub_)}'] = ','
    for _, row in user_publisher_stat.iterrows():
        user_id = row['user_id']
        publisher = row['publisher']
        users.loc[users['user_id'] == user_id, f'publisher_{int(publisher)}'] = str(row['log_cpm'])[1:-1]
    for pub_ in history.publisher.unique():
        users[f'publisher_{int(pub_)}'] = users[f'publisher_{int(pub_)}'].apply(
            lambda x: [float(i) for i in x.split(',') if i != ''])

    val['user_ids'] = val['user_ids'].apply(lambda x: [int(i) for i in x.split(',')])
    val['publishers'] = val['publishers'].apply(lambda x: [int(i) for i in x.split(',')])
    val['duration_hours'] = val['hour_end'] - val['hour_start']

    ans = pd.DataFrame()
    ans['at_least_one'] = val.apply(lambda x: calc(x, hours24, history, users), axis=1)
    ans['at_least_two'] = val.apply(lambda x: calc2(x, hours24, history, users), axis=1)
    ans['at_least_three'] = val.apply(lambda x: calc3(x, hours24, history, users), axis=1)

    ans[['at_least_one', 'at_least_two', 'at_least_three']].to_csv("validate_answers_true.tsv",
                                                                   sep="\t", index=False, header=True)


if __name__ == '__main__':
    main()
