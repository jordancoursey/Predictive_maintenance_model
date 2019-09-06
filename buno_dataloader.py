import numpy as np
import os
import pandas as pd
import torch

mission_feat_dir = 'buno-data/buno-training-set/buno-mission-feat'
maint_feat_dir = 'buno-data/buno-training-set/buno-maint-feat'
break_history_fp = 'buno-data/buno-training-set/break-history.csv'
cum_fh_dir = 'buno-data/buno-cum-fh-files'

mission_exclude_cols = ['Buno', 'Bu/SerNo', 'Date', 'LaunchDate', 'dam', 'Nan', 'Cum FH']
maint_exclude_cols = ['Date', 'Buno', 'Cum FH']

class BunoDataset:
    def __init__(self, bunos, shuffle=True):
        self.bunos = np.asarray(bunos)
        self.break_history = pd.read_csv(break_history_fp)
        self.break_history = self.break_history[self.break_history['Buno'].isin(bunos)]
        self.break_history['Date'] = pd.to_datetime(self.break_history['Date'])
        self.len = self.break_history.shape[0]
        self.shuffle = shuffle
        self.mission_feat_dim, self.maint_feat_dim = self.get_feat_dims()

    def __len__(self):
        return self.len

    def __iter__(self):
        # Shuffle the BUNOs and breakage records
        if self.shuffle:
            self.bunos = np.random.permutation(self.bunos)
            self.break_history = self.break_history.sample(frac=1)
            self.break_history = self.break_history.reset_index(drop=True)

        for buno in self.bunos:
            mission_fp, maint_fp, cum_fh_fp = self.get_buno_filepaths(buno)
            mission_history = pd.read_csv(mission_fp, low_memory=False)
            maint_history = pd.read_csv(maint_fp, low_memory=False)
            buno_break_history = self.break_history[self.break_history['Buno'] == buno].reset_index()
            buno_cum_fh = pd.read_csv(cum_fh_fp)

            # Convert dates to pd.DateTime objects
            mission_history['LaunchDate'] = pd.to_datetime(mission_history['LaunchDate'])
            maint_history['Date'] = pd.to_datetime(maint_history['Date'])
            buno_cum_fh['Date'] = pd.to_datetime(buno_cum_fh['Date'])

            for i, row in buno_break_history.iterrows():
                current_date = row['Date']
                mission_query = (mission_history['LaunchDate'] < current_date)
                maint_query = (maint_history['Date'] < current_date)
                missions = mission_history[mission_query]
                maint = maint_history[maint_query]

                if maint.isnull().values.any():
                    continue

                # Get the amount of flight hours that have passed since each event
                current_fh = buno_cum_fh[buno_cum_fh['Date'] == current_date].iloc[0]['Cum FH']
                # missions['FH Delta'] = current_fh - missions['Cum FH']
                # maint['FH Delta'] = current_fh - maint['Cum FH']
                # maint = maint.fillna(0)

                mission_features = [col for col in missions.columns if col not \
                                    in mission_exclude_cols]
                maint_features = [col for col in maint.columns if col not \
                                  in maint_exclude_cols]

                x_missions = np.asarray(missions[mission_features])
                x_maint = np.asarray(maint[maint_features])
                y = row['dam']

                yield x_missions, x_maint, y

    def get_buno_filepaths(self, buno):
        mission_fname = f'{str(buno)}-mission-feat.csv'
        maint_fname = f'{str(buno)}-maint-feat.csv'
        cum_fh_fname = f'{str(buno)}-cum-fh.csv'
        mission_fp = os.path.join(mission_feat_dir, mission_fname)
        maint_fp = os.path.join(maint_feat_dir, maint_fname)
        cum_fh_fp = os.path.join(cum_fh_dir, cum_fh_fname)

        return mission_fp, maint_fp, cum_fh_fp

    def get_feat_dims(self):
        mission_fp, maint_fp, _ = self.get_buno_filepaths(self.bunos[0])

        missions = pd.read_csv(mission_fp)
        maint = pd.read_csv(maint_fp)

        mission_drop_cols = [col for col in missions.columns if col \
                            in mission_exclude_cols]
        maint_drop_cols = [col for col in maint.columns if col \
                          in maint_exclude_cols]

        mission_feat_dim = missions.drop(mission_drop_cols, axis=1).shape[1]
        maint_feat_dim = maint.drop(maint_drop_cols, axis=1).shape[1]

        return mission_feat_dim, maint_feat_dim

class BunoDataloader:
    def __init__(self, buno_dataset, batch_size):
        self.buno_dataset = buno_dataset
        self.batch_size = batch_size
        self.len = int(np.ceil(len(buno_dataset) / batch_size))

    def __iter__(self):
        x_mission_batch, x_maint_batch, y_batch = [], [], []

        for i, (x_mission, x_maint, y) in enumerate(self.buno_dataset):
            if len(y_batch) == self.batch_size:
                yield x_mission_batch, x_maint_batch, y_batch

                x_mission_batch, x_maint_batch, y_batch = [], [], []
            elif i == len(self.buno_dataset) - 1:
                yield x_mission_batch, x_maint_batch, y_batch
            else:
                try:
                    x_mission_batch.append(torch.tensor(x_mission).float())
                    x_maint_batch.append(torch.tensor(x_maint).float())
                    y_batch.append(torch.tensor(y).float())
                except TypeError:
                    pass

    def __len__(self):
        return self.len

    @property
    def mission_feat_dim(self):
        return self.buno_dataset.mission_feat_dim

    @property
    def maint_feat_dim(self):
        return self.buno_dataset.maint_feat_dim

# buno_dataset = BunoDataset([165905])
# dataloader = BunoDataloader(buno_dataset, batch_size=2)

# for i, (x_mission, x_maint, y) in enumerate(buno_dataset):
#     print(x_maint)
#     if i == 1:
#         break

# for i, (x_mission_batch, x_maint_batch, y_batch) in enumerate(dataloader):
#     print(y_batch)
#     print(torch.cat(x_mission_batch).size(), torch.cat(x_maint_batch).size(), torch.tensor(y_batch).size())
