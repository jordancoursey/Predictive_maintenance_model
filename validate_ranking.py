from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
import torch
from torch.nn import Linear
from torch import sigmoid

from rank_bunos import rank_bunos

parser = ArgumentParser(prog='rank_bunos.py',
                        description='Takes as input a list of BUNOs and a date ' \
                        'and ranks them according to their likelihoods of staying MC.')
parser.add_argument('-b', '--bunos', type=str, dest='buno_fp', default=None,
                    help='The relative filepath of a txt file containing the ' \
                    'BUNOs that you would like to see ranked.')
parser.add_argument('-d', '--date', type=str, dest='date', default=None,
                    help='The date on which to rank the BUNOs. Must be in the ' \
                    'format DD/MM/YYYY.')
parser.add_argument('-m', '--model', type=str, dest='model_fp', default=None,
                    help='The relative filepath of a file containing a saved ' \
                    'PyTorch model.')
args = parser.parse_args()

mission_exclude_cols = ['Buno', 'Bu/SerNo', 'Date', 'LaunchDate', 'dam', 'Nan', 'Cum FH']
maint_exclude_cols = ['Date', 'Buno', 'Cum FH']

mission_dir

mission_dir = 'buno_files/buno_mission'
maint_dir = 'buno_files/buno_maint'

buno_fp = args.buno_fp
model_fp = args.model_fp
pred_date = pd.to_datetime(args.date)

with open(buno_fp, 'r') as f:
    bunos = [line.strip('\n') for line in f.readlines()]

ranked_bunos = rank_bunos(bunos, pred_date, model_fp)

# Get the true order by finding out which BUNOs actually broke first
buno_break_dates = dict()

for buno in bunos:
    mission_fp = os.path.join(mission_dir, f'{buno}-mission-feat.csv')
    buno_df = pd.read_csv(mission_fp)
    buno_df['LaunchDate'] = pd.to_datetime(buno_df['LaunchDate'])

    break_date = None

    after_pred_date = buno_df[buno_df['LaunchDate'] >= pred_date]

    if len(after_pred_date):
        break_dates = after_pred_date[after_pred_date['dam'] == 1]
        if len(break_dates):
            break_date = break_dates['LaunchDate'].iloc[0]

    if break_date:
        buno_break_dates[buno] = break_date
    else:
        buno_break_dates[buno] = pd.Timestamp.max

rank = lambda x: x[1]
actual_break_dates = list(buno_break_dates.items())
actual_ranking = sorted(actual_break_dates, key=rank)[::-1]

print('Ranking of likelihood to remaint MC according to model:')

for i, ranked_buno in enumerate(ranked_bunos[::-1]):
    print(str(i + 1) + '.', ranked_buno[0], 1 - ranked_buno[1])

print('\nTrue ranking according to first BUNO to break:')

for i, ranked_buno in enumerate(actual_ranking[::-1]):
    print(str(i + 1) + '.', ranked_buno[0])

print('\n')
