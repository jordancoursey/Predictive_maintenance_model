from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
import torch
from torch.nn import Linear
from torch import sigmoid

mission_dir = 'buno_files/buno_mission'
maint_dir = 'buno_files/buno_maint'
cum_fh_dir = 'buno-cum-fh-files'

mission_exclude_cols = ['Buno', 'Bu/SerNo', 'Date', 'LaunchDate', 'dam', 'Nan', 'Cum FH']
maint_exclude_cols = ['Date', 'Buno', 'Cum FH']

def rank_bunos(bunos, pred_date, model_fp):
# Load the mission and maintenance histories for each BUNO
    features = dict()

    for buno in bunos:
        mission_fp = os.path.join(mission_dir, f'{buno}-mission-feat.csv')
        maint_fp = os.path.join(maint_dir, f'{buno}-maint-feat.csv')
        cum_fh_fp = os.path.join(cum_fh_dir, f'{buno}-cum-fh.csv')

        mission_df = pd.read_csv(mission_fp)
        maint_df = pd.read_csv(maint_fp)
        cum_fh_df = pd.read_csv(cum_fh_fp)
        mission_df['LaunchDate'] = pd.to_datetime(mission_df['LaunchDate'])
        maint_df['Date'] = pd.to_datetime(maint_df['Date'])
        cum_fh_df['Date'] = pd.to_datetime(cum_fh_df['Date'])

        # Get all of the mission and maintenance that happened before the prediction
        # date
        mission_hist = mission_df[mission_df['LaunchDate'] < pred_date]
        maint_hist = maint_df[maint_df['Date'] < pred_date]

        # Get the timedelta feature
        # current_fh = cum_fh_df[cum_fh_df['Date'] == pred_date].iloc[0]['Cum FH']
        # mission_hist['FH Delta'] = current_fh - mission_hist['Cum FH']
        # maint_hist['FH Delta'] = current_fh - maint_hist['Cum FH']
        mission_hist = mission_hist.fillna(0)
        maint_hist = maint_hist.fillna(0)

        # Make a list of the features we want to keep
        mission_features = [col for col in mission_hist.columns if col not \
                            in mission_exclude_cols]
        maint_features = [col for col in maint_hist.columns if col not \
                          in maint_exclude_cols]

        x_mission = np.asarray(mission_hist[mission_features])
        x_maint = np.asarray(maint_hist[maint_features])
        x_mission, x_maint = torch.tensor(x_mission).float(), torch.tensor(x_maint).float()

        features[buno] = (x_mission, x_maint)

        mission_dim = x_mission.shape[1]
        maint_dim = x_maint.shape[1]

    # Load the model
    mission_model = Linear(mission_dim, 1)
    maint_model = Linear(maint_dim, 1)

    ckpt = torch.load(model_fp, map_location='cpu')
    mission_model.load_state_dict(ckpt['mission_state_dict'])
    maint_model.load_state_dict(ckpt['maint_state_dict'])

    # Predict breakage probabilities for each BUNO
    mission_model.eval()
    maint_model.eval()
    breakage_probs = dict()

    with torch.no_grad():
        for buno in bunos:
            x_mission, x_maint = features[buno]

            mission_logits = mission_model(x_mission)
            maint_logits = maint_model(x_maint)

            logit = torch.sum(mission_logits) + torch.sum(maint_logits)
            pred_prob = sigmoid(logit)

            breakage_probs[buno] = pred_prob.detach().numpy()

    # Now rank the BUNOs
    rank = lambda x: 1 - x[1]
    ranked_bunos = sorted(breakage_probs.items(), key=rank)

    return ranked_bunos

if __name__ == '__main__':
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

    buno_fp = args.buno_fp
    model_fp = args.model_fp
    pred_date = pd.to_datetime(args.date)

    with open(buno_fp, 'r') as f:
        bunos = [line.strip('\n') for line in f.readlines()]

    ranked_bunos = rank_bunos(bunos, pred_date, model_fp)

    # Display the ranking
    print('\nDisplay Buno (left) and probability of remaining MC (right)')

    for i, buno in enumerate(ranked_bunos):
        print(str(i + 1) + '.', buno[0], 1 - buno[1])

    print('\n')
