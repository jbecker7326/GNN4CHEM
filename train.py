import time
import torch
import metrics
#import torchaudio
from torchmetrics import F1Score
import numpy as np

def train(model, device, train_loader, val_loader, in_channels, out_channels, num_epochs, lr, debug=False):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    f1 = F1Score(task="multilabel", num_labels = out_channels).to(device)

    losses, f_score, val_losses, val_f_score = [], [], [], []
    t0, time_elapsed = time.time(), []
    
    for epoch in range(num_epochs):
        total_loss, batch_count, f_total = 0, 0, 0
        val_total_loss, val_batch_count, val_f_total = 0, 0, 0

        # training
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch.to(device))
            loss = loss_fn(pred, batch.y.to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            f_total += f1(pred, batch.y)
         
        mean_loss = total_loss / batch_count
        mean_f1 = f_total / batch_count
        losses.append(mean_loss)
        f_score.append(mean_f1)
        
        if debug:
            print(f"train loss at epoch {epoch} = {mean_loss}")
            print(f"train mean f1 score at epoch {epoch} = {mean_f1}")
            
        # validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch.to(device))
                loss = loss_fn(output, batch.y.to(device))
                val_total_loss += loss.item()
                val_batch_count += 1
                val_f_total += f1(output,batch.y)
    
        val_mean_loss = val_total_loss / val_batch_count
        val_mean_f1 = val_f_total / val_batch_count
        val_losses.append(val_mean_loss)
        val_f_score.append(val_mean_f1)
        time_elapsed.append(time.time()-t0)
        
        if debug:
            print(f"validation loss at epoch {epoch} = {val_mean_loss}")
            print(f"validation mean f1 score at epoch {epoch} = {val_mean_f1}")

    # run statistics
    stats = [mean_loss, mean_f1, val_mean_loss, val_mean_f1, time_elapsed[-1]]

    # add cuda to statistics if used
    if device == 'cuda':
        stats = stats + [torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)]
    else:
        stats = stats + [np.nan, np.nan]


    return model, losses, val_losses, stats
