## imports
#import torchaudio
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
import torch
import torch.nn as nn
from torch_geometric.nn import MFConv, TransformerConv, ClusterGCNConv

from train import train
from test import test

import gc
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## read in command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-ne", "--num_epochs", nargs='?', default=8000, type=int, help="number of epochs")
parser.add_argument("-lr", "--learning_rate", nargs='?', default=1e-3, type=float, help="optimizer learning rate")
parser.add_argument("-ns", "--num_smooth", nargs='?', default=1, type=int, help="smoothing factor for plots")
parser.add_argument("-nj", "--num_jobs", nargs='?', default=5, type=int, help="number of jobs for cross validation")
parser.add_argument("-bs", "--batch_size", nargs='?', default=8, type=int, help="batch size for training data loader")
parser.add_argument("-conn", "--sql_connection", nargs='?', default=None, type=str, help="path for sqlite database connection. will create and save to database if one is specified.")
parser.add_argument("-read_db", "--read_db", nargs='?', default=False, type=bool, help="trigger to read from sql database instead of running models. will not run new models if set to true.")
args = parser.parse_args()


## looper for running separate instances of each model over multiple runs for cross validation
def looper(import_name, model_name, train_loader, val_loader, test_loader, in_channels, out_channels, debug=False, num_epochs=8000, lr=1e-3, njobs=5, conn=None):
    losses, val_losses, train_stats, test_stats = [], [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for job in range(njobs):
        print(f'running {model_name} job {job}...')
        model = import_name.GCNConvNet(in_channels=in_channels, out_channels=out_channels)
        model, loss, val_loss, train_stat = train(model, device, train_loader, val_loader, in_channels, out_channels, num_epochs, lr, debug)
        test_stat = test(model, device, test_loader, out_channels)
    
        losses.append(loss)
        val_losses.append(val_loss)
        train_stats.append(train_stat)
        test_stats.append(test_stat)

        # clear cuda cache for accurate memory analysis
        if device == "cuda":
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # move from cuda to cpu if needed
        train_stats = [[x.item() if torch.is_tensor(x) else x for x in l] for l in train_stats]
        test_stats = [[x.item() if torch.is_tensor(x) else x for x in l] for l in test_stats]

    # average over jobs  
    losses, val_losses, train_stats, test_stats = np.mean(losses,axis=0), np.mean(val_losses,axis=0), np.mean(train_stats,axis=0), np.mean(test_stats,axis=0)
    

    # save to sqlite database if specified
    if conn:
        pd.DataFrame(losses).to_sql(f'{model_name}_losses_ne{num_epochs}_lr{lr}_nj{njobs}', conn, if_exists='replace', index=False)
        pd.DataFrame(val_losses).to_sql(f'{model_name}_val_losses_ne{num_epochs}_lr{lr}_nj{njobs}', conn, if_exists='replace', index=False)
        pd.DataFrame(train_stats).to_sql(f'{model_name}_train_stats_ne{num_epochs}_lr{lr}_nj{njobs}', conn, if_exists='replace', index=False)
        pd.DataFrame(test_stats).to_sql(f'{model_name}_test_stats_ne{num_epochs}_lr{lr}_nj{njobs}', conn, if_exists='replace', index=False)
    
    return losses, val_losses, train_stats, test_stats


def main():
    lr=args.learning_rate
    num_epochs=args.num_epochs
    num_smooth=args.num_smooth
    njobs = args.num_jobs
    batch_size = args.batch_size
    conn = args.sql_connection
    
    train_dataset = datasets.PPI(root="./data", split="train")
    train_dataset.shuffle()
    val_dataset = datasets.PPI(root="./data", split="val")
    test_dataset = datasets.PPI(root="./data", split="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    in_channels = train_dataset.num_features
    out_channels = train_dataset.num_classes

    train_stats_all = []
    test_stats_all = []
    comb_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200, layout='tight')

    if conn:
        conn = sqlite3.connect(conn)

    # model names and paths to model .py files
    models = {'Cluster': 'models.JB_Cluster',
              'GAT': 'models.GNN_GAT',
              'GIN': 'models.GNN_GIN',
              'MF': 'models.JB_MF',
              'MF+Cluster': 'models.JB_ClusterMF',
              'MF+GIN': 'models.JB_MFGIN',
              'MF+Transformer': 'models.JB_MFT',
              'SAGE': 'models.GNN_GS'
             }
    
    for model_name, model_path in models.items():
        print(model_name)
        # read from sqlite database instead of running model
        if args.read_db:
            loss = pd.read_sql_query(f"select * from '{model_name}_losses_ne{num_epochs}_lr{lr}_nj{njobs}'", conn).to_numpy()
            val_loss = pd.read_sql_query(f"select * from '{model_name}_val_losses_ne{num_epochs}_lr{lr}_nj{njobs}'", conn).to_numpy()
            train_stats = pd.read_sql_query(f"select * from '{model_name}_train_stats_ne{num_epochs}_lr{lr}_nj{njobs}'", conn).to_numpy().squeeze(1)
            test_stats = pd.read_sql_query(f"select * from '{model_name}_test_stats_ne{num_epochs}_lr{lr}_nj{njobs}'", conn).to_numpy().squeeze(1)

        # run model
        else:
            model = __import__(model_path, fromlist=[None])
            loss, val_loss, train_stats, test_stats = looper(model, model_name, train_loader, val_loader, test_loader, in_channels, out_channels, num_epochs=num_epochs, lr=lr, njobs = njobs, conn=conn)   

        # individual model learning curves
        fig, ax = plt.subplots()
        ax.plot(range(num_epochs), loss, color='orange', label='Training')
        ax.plot(range(num_epochs), val_loss, color='darkblue', label='Validation')
        ax.set_title(f'{model_name} Loss Curve')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid()
        print(f'exporting {model_name} learning curve to images/{model_name}_learning_curve_ne{num_epochs}_lr{lr}_nj{njobs}.png')
        fig.savefig(f'images/{model_name}_learning_curve_ne{num_epochs}_lr{lr}_nj{njobs}.png')
    
        # combined loss curve plots
        ax1.plot(range(0,num_epochs,num_smooth), loss.tolist()[0::num_smooth], label=model_name)
        ax2.plot(range(0,num_epochs,num_smooth), val_loss.tolist()[0::num_smooth], label=model_name)
        # statistic exports
        train_stats_all.append(train_stats)
        test_stats_all.append(test_stats)

    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax1.set_xlim(0,num_epochs)
    ax1.set_ylim(150,180)
    
    ax2.set_title('Validation Loss Curve')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid()
    ax2.set_xlim(0,num_epochs)
    ax2.set_ylim(150,180)

    hyper_string = f'ne{num_epochs}_lr{lr}_nj{njobs}'
    print(f'exporting combined loss plot to images/combined_learning_curve_{hyper_string}.png')
    comb_fig.savefig(f'images/combined_learning_curve_{hyper_string}.png')

    train_stats_all = pd.DataFrame(train_stats_all)
    train_stats_all.columns = ['train_loss', 'train_mean_f1', 'val_loss', 'val_mean_f1', 'time', 'mem_alloc', 'mem_res']
    train_stats_all.insert(0,column='model', value=models.keys())
    print(f'exporting model train statistics to train_stats_{hyper_string}.csv')
    train_stats_all.to_csv(f'train_stats_ne{num_epochs}_{hyper_string}.csv')
    
    test_stats_all = pd.DataFrame(test_stats_all)
    #test_stats_all.columns = ['MSE', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'MCC', 'AUROC', 'AUPRC']
    test_stats_all.columns = ['MSE', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'AUROC', 'AUPRC']
    test_stats_all.insert(0,column='model', value=models.keys())
    print(f'exporting model test statistics to test_stats_{hyper_string}.csv')
    test_stats_all.to_csv(f'test_stat_{hyper_string}.csv')


if __name__ == '__main__':
    main()