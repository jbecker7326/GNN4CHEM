import metrics
import torch
from models import GNN_my_model, GNN_GS
import networkx as nx
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from torch_geometric.utils import to_networkx
from torchmetrics import F1Score

def test(test_loader, in_channels, out_channels, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GNN_GS.GSage(in_channels=in_channels, out_channels=out_channels)
    model = GNN_my_model.GCNConvNet(in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(torch.load(f"model_state/ysl_{lr}_epoch_{num_epochs}.pth")) #path to load the model
    model.to(device)
    model.eval()
    f1 = F1Score(task="multilabel", num_labels = out_channels).to(device)
    predictions = torch.Tensor()
    labels = torch.Tensor()

    with torch.no_grad():
        for batch in test_loader:
          batch = batch.to(device)
          output = model(batch.to(device))
          predictions = torch.cat((predictions, output.cpu()), 0)
          labels = torch.cat((labels, batch.y.to(device)), 0)
    labels = labels.numpy().flatten()
    predictions = predictions.numpy().flatten()


    mse_loss = metrics.get_mse(labels, predictions)
    acc = metrics.get_accuracy(labels, predictions, 0.5)
    prec = metrics.precision(labels, predictions, 0.5)
    sensitivity = metrics.sensitivity(labels, predictions,  0.5)
    specificity = metrics.specificity(labels, predictions, 0.5)
    f1 = metrics.f_score(labels, predictions, 0.5)
    mcc = metrics.mcc(labels, predictions,  0.5)
    auroc = metrics.auroc(labels, predictions)
    auprc = metrics.auprc(labels, predictions)


    print(f'MSE loss : {mse_loss}')
    print(f'Accuracy : {acc}')
    print(f'precision: {prec}')
    print(f'Sensititvity :{sensitivity}')
    print(f'specificity : {specificity}')
    print(f'f-score : {f1}')
    print(f'MCC : {mcc}')
    print(f'AUROC: {auroc}')
    print(f'AUPRC: {auprc}')