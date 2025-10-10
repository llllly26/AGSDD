import argparse
import sys
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch,Data
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from tqdm.auto import tqdm
from ema_pytorch import EMA

from dataset_src.large_dataset import Cath
from main import EGNN_NET,AGSDD, seq_recovery

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int,default=16,help='batch_size')
parser.add_argument('--step', type = int,default=500,help='step')
parser.add_argument('--ensemble_num', type = int,default=1,help='ensemble_num')
parser.add_argument('--seed', type = int,default=42,help='seed')
args = parser.parse_args()
set_seed(args.seed)

ckpt = torch.load('/mnt/disks/Align_repr/results/weight/BestModel.pt')

config=ckpt['config']

gnn = EGNN_NET(input_feat_dim=config['input_feat_dim'],hidden_channels=config['hidden_dim'],edge_attr_dim=config['edge_attr_dim'],dropout=config['drop_out'],n_layers=config['depth'],update_edge = config['update_edge'],embedding=config['embedding'],embedding_dim=config['embedding_dim'],embed_ss=config['embed_ss'],norm_feat=config['norm_feat'])
diffusion = AGSDD(model=gnn, config=config)
diffusion = EMA(diffusion, beta = 0.995, update_every = 10)
diffusion.load_state_dict(ckpt['ema'])

test_dir = '/mnt/disks/wcls/Datas/Norm_TS500_N_30_CATHtrain_Norm/TS50/test/'

test_ID = os.listdir(test_dir)
test_dataset = Cath(test_ID, test_dir)
test_dataset = test_dataset
bs = args.batch_size
print(f"test_dataset len:{len(test_dataset)}")

test_loader = DataLoader(test_dataset,batch_size=bs, shuffle=False, pin_memory = False, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

diffusion.ema_model.eval()
diffusion.to(device)
perplexity, recovery_list = [], []
ensemble_num = args.ensemble_num

with torch.no_grad():
    ind_all = torch.tensor([])  # cpu.
    all_probs = torch.tensor([])
    all_seq = torch.tensor([])
    for data in test_loader:
        data = data.to(device)
    
        all_prob = []
        # all_conf = []
        for i in tqdm(range(ensemble_num)):
            prob, sample_graph = diffusion.ema_model.ddim_sample(data, diverse=True, step=args.step)   # True
            all_prob.append(prob)

        all_zt_tensor = torch.stack(all_prob)
        ind = (all_zt_tensor.mean(dim=0).argmax(dim=1) == data.x.argmax(dim=1))
        probs = all_zt_tensor.mean(dim=0)
        
        ind_all = torch.cat([ind_all, ind.cpu()])
        all_probs = torch.cat([all_probs, probs.cpu()])
        all_seq = torch.cat([all_seq, data.x.cpu()])
        # break
    
    recovery = (ind_all.sum() / ind_all.shape[0]).item() # float.
    ll_fullseq = F.cross_entropy(all_probs, all_seq, reduction='mean')
    perplexity.append(np.exp(ll_fullseq))
    print(f"recovery is: {recovery:.6f}")
    print(f"perplexity is: {perplexity[-1]:.3f}")