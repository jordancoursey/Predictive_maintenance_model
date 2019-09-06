from argparse import ArgumentParser
from datetime import datetime as dt
from datetime import timedelta
from itertools import chain
import numpy as np
import os
import torch
from torch.nn import BCEWithLogitsLoss, Linear
from torch.optim import Adam, LBFGS
from yaml import load as load_yaml

from buno_dataloader import BunoDataloader, BunoDataset

parser = ArgumentParser()
parser.add_argument('-c', '--continue', type=str, dest='continue_fname', default=None)
parser.add_argument('-s', '--save', type=str, dest='save_fname', default=None)
args = parser.parse_args()

def get_flat_params(network):
    return torch.cat([param.view(-1) for param in network.parameters()])

def evaluate(mission_model, maint_model, loader, device):
    mission_model.eval()
    maint_model.eval()

    n_correct = 0
    n_seen = 0

    n = len(val_loader)

    with torch.no_grad():
        for i, (x_mission, x_maint, y) in enumerate(loader):
            print(f'{i + 1}/{n}')
            x_mission = torch.cat(x_mission_batch, dim=0)
            x_maint = torch.cat(x_maint_batch, dim=0)
            y = torch.tensor(y_batch)

            x_mission = x_mission.to(device)
            x_maint = x_maint.to(device)
            y = y.to(device)

            mission_logits_daily = mission_model(x_mission)
            maint_logits_daily = maint_model(x_maint)

            mission_logits = torch.stack([torch.sum(mission_logits_daily[slice]) \
                              for slice in mission_hist_slices])
            maint_logits = torch.stack([torch.sum(maint_logits_daily[slice]) \
                            for slice in maint_hist_slices])
            logits = mission_logits + maint_logits
            preds = (logits > 0.0)

            n_correct += torch.sum(preds == (y == 1.0))
            n_seen += y.size(0)

    acc = (n_correct / n_seen).detach().cpu().numpy()
    print(f'VAL ACC: {acc}')

    return acc

config_path = 'config.yaml'
save_dir = 'saved-sessions'

bunos = np.load('miscellaneous/bunos.npy')
train_bunos = bunos[5:-5]
val_bunos = bunos[:1]
config = load_yaml(open(config_path, 'r'))

train_batch_size = config['train_batch_size']
val_batch_size = config['val_batch_size']
val_every = config['val_every']
save_every = config['save_every']
lr = config['learning_rate']
l2_reg_coef = config['l2_reg_coef']
continue_fname = args.continue_fname
save_fname = args.save_fname

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

save_path = os.path.join(save_dir, save_fname)

if torch.cuda.is_available():
    print('Using CUDA.\n')
    device = torch.device('cuda')
else:
    print('CUDA is unavailable. Using CPU.\n')
    device = torch.device('cpu')

print('Preparing training/validation loaders...')
train_loader = BunoDataloader(BunoDataset(train_bunos), train_batch_size)
val_loader = BunoDataloader(BunoDataset(val_bunos), val_batch_size)
print('Done!\n')

mission_feat_dim = train_loader.mission_feat_dim
maint_feat_dim = train_loader.maint_feat_dim

mission_model = Linear(in_features=mission_feat_dim, out_features=1)
mission_model.weight.data *= 0.0
mission_model.bias.data *= 0.0
maint_model = Linear(in_features=maint_feat_dim, out_features=1)
maint_model.weight.data *= 0.0
maint_model.bias.data *= 0.0
mission_model.to(device)
maint_model.to(device)

if continue_fname:
    load_path = os.path.join('saved-sessions', continue_fname)
    print(f'Loading saved session from {load_path}...')

    ckpt = torch.load(load_path)
    mission_model.load_state_dict(ckpt['mission_state_dict'])
    maint_model.load_state_dict(ckpt['maint_state_dict'])
    train_losses = ckpt['train_losses']
    val_accs = ckpt['val_accs']
    time_elapsed = ckpt['time_elapsed']

    print('Done!\n')

print('Training regression model...\n')

loss_fun = BCEWithLogitsLoss()
optimizer = LBFGS(chain(mission_model.parameters(), maint_model.parameters()), lr=lr)
# optimizer = Adam(chain(mission_model.parameters(), maint_model.parameters()), lr=lr)

n_seen = 0
start_time = dt.now()
time_elapsed = timedelta(0)
train_losses = []
val_accs = []

for i, (x_mission_batch, x_maint_batch, y_batch) in enumerate(train_loader):
    mission_model.train()
    maint_model.train()

    mission_limit_indices = [0] + [x.size(0) for x in x_mission_batch]
    maint_limit_indices = [0] + [x.size(0) for x in x_maint_batch]
    mission_hist_slices = [slice(start, end) for start, end in zip(mission_limit_indices[:-1], mission_limit_indices[1:])]
    maint_hist_slices = [slice(start, end) for start, end in zip(maint_limit_indices[:-1], maint_limit_indices[1:])]

    x_mission = torch.cat(x_mission_batch, dim=0)
    x_maint = torch.cat(x_maint_batch, dim=0)
    y = torch.tensor(y_batch)

    x_mission = x_mission.to(device)
    x_maint = x_maint.to(device)
    y = y.to(device)

    loss_reported = False

    def bce_loss():
        optimizer.zero_grad()

        mission_logits_daily = mission_model(x_mission)
        maint_logits_daily = maint_model(x_maint)

        mission_logits = torch.stack([torch.sum(mission_logits_daily[slice]) \
                          for slice in mission_hist_slices])
        maint_logits = torch.stack([torch.sum(maint_logits_daily[slice]) \
                        for slice in maint_hist_slices])
        logits = mission_logits + maint_logits

        print(torch.sigmoid(logits))

        # bce = loss_fun(logits, y)
        bce = loss_fun(mission_logits, y)

        flat_params = torch.cat([get_flat_params(mission_model),
                                 get_flat_params(maint_model)])
        l2_loss = l2_reg_coef * torch.sum(torch.pow(flat_params, 2))
        reg_loss = bce + l2_loss

        reg_loss.backward()

        return reg_loss

    optimizer.step(bce_loss)

    with torch.no_grad():
        mission_logits_daily = mission_model(x_mission)
        maint_logits_daily = maint_model(x_maint)

        mission_logits = torch.stack([torch.sum(mission_logits_daily[slice]) \
                          for slice in mission_hist_slices])
        maint_logits = torch.stack([torch.sum(maint_logits_daily[slice]) \
                        for slice in maint_hist_slices])
        logits = mission_logits + maint_logits

        bce = loss_fun(logits, y)
        bce = loss_fun(mission_logits, y)
        train_losses.append(bce.cpu().detach().numpy())

    n_seen += 1
    time_elapsed += dt.now() - start_time

    update_str = 'Elapsed Time: {0}, Batches Seen: {1}/{2}, Avg. Loss: {3:.4f}'
    elapsed_time_str = ''.join(str(time_elapsed).split('.')[0])
    format_args = (elapsed_time_str, n_seen, len(train_loader), np.mean(train_losses[-5:]))
    print(update_str.format(*format_args))

    if not n_seen % val_every:
        val_acc = evaluate(mission_model, maint_model, val_loader, device)
        val_accs.append(val_acc)

    if not n_seen % save_every:
        ckpt = dict(mission_state_dict=mission_model.state_dict(),
                    maint_state_dict=maint_model.state_dict(),
                    train_losses=train_losses,
                    val_accs=val_accs,
                    time_elapsed=time_elapsed)
        torch.save(ckpt, save_path)

print('\nTraining complete!\n')
