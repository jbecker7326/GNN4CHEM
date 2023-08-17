import metrics
import torch
#import networkx as nx
import numpy as np
#from IPython.display import HTML
#from matplotlib import animation
#from torch_geometric.utils import to_networkx
from torchmetrics import F1Score, Accuracy, Precision, Specificity, Recall, MatthewsCorrCoef, MeanSquaredError, AUROC, PrecisionAtFixedRecall, AveragePrecision

def test(model, device, test_loader, out_channels, debug=False):
    model.to(device)
    model.eval()
    predictions = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    
    with torch.no_grad():
        for batch in test_loader:
          batch = batch.to(device)
          output = model(batch.to(device))
          predictions = torch.cat((predictions, output.to(device)), 0)
          labels = torch.cat((labels, batch.y.to(device)), 0)
    
    f1score = F1Score(task="multilabel", num_labels = out_channels).to(device)
    msescore = MeanSquaredError().to(device)
    accuracy = Accuracy(task="multilabel", num_labels = out_channels).to(device)
    precision = Precision(task="multilabel", num_labels = out_channels).to(device)
    specificity = Specificity(task="multilabel", num_labels = out_channels).to(device)
    sensitivity = Recall(task="multilabel", num_labels = out_channels).to(device)
    #matthews_corrcoef = MatthewsCorrCoef(task="multilabel", num_labels = out_channels).to(device)
    auroc = AUROC(task="multilabel", num_labels = out_channels).to(device)
    auprc = AveragePrecision(task="multilabel", num_labels = out_channels).to(device)

    #stats = [msescore(predictions, labels), accuracy(predictions, labels), precision(predictions, labels), sensitivity(predictions, labels), specificity(predictions, labels), f1score(predictions, labels), matthews_corrcoef(predictions, labels), auroc(predictions, labels.int()), auprc(predictions, labels.int())]
    stats = [msescore(predictions, labels), accuracy(predictions, labels), precision(predictions, labels), sensitivity(predictions, labels), specificity(predictions, labels), f1score(predictions, labels), auroc(predictions, labels.int()), auprc(predictions, labels.int())]

    if debug:
        print(f'MSE loss : {stats[0]}')
        print(f'Accuracy : {stats[1]}')
        print(f'precision: {stats[2]}')
        print(f'Sensititvity : {stats[3]}')
        print(f'specificity : {stats[4]}')
        print(f'f-score : {stats[5]}')
        #print(f'MCC : {stats[6]}')
        print(f'AUROC: {stats[6]}')
        print(f'AUPRC: {stats[7]}\n')

    return stats
