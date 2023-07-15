import time
import torch
from models import GNN_my_model
import matplotlib.pyplot as plt
import metrics

def train(train_loader, num_epochs, in_channels, out_channels, lr, val_loader, print=False):
    # model training
    model = GNN_my_model.GCNConvNet(in_channels=in_channels, out_channels=out_channels)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(my_device)
    losses = []
    f_score = []
    val_losses = []
    val_f_score = []
    time_elapsed = []
    epochs = []
    val_epochs = []
    t0 = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        f_total = 0
        val_f_total = 0
        val_total_loss = 0
        val_batch_count = 0
        # training
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch.to(my_device))
            loss = loss_fn(pred, batch.y.to(my_device))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            batch_count += 1
            f_total += metrics.f_score(batch.y.to(my_device).numpy().flatten(), pred.detach().numpy().flatten(), 0.5)
        mean_loss = total_loss / batch_count
        mean_f1 = f_total / batch_count
        losses.append(mean_loss)
        epochs.append(epoch)
        f_score.append(mean_f1)
        if print:
            print(f"train loss at epoch {epoch} = {mean_loss}")
            print(f"train mean f1 score at epoch {epoch} = {mean_f1}")
        # validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(my_device)
                output = model(batch.to(my_device))
                loss = loss_fn(output, batch.y.to(my_device))
                val_total_loss += loss.item()
                val_batch_count += 1
                val_f_total += metrics.f_score(batch.y.to(my_device).numpy().flatten(), output.detach().numpy().flatten(),
                                           0.5)
        mean_loss = val_total_loss / val_batch_count
        mean_f1 = val_f_total / val_batch_count
        val_losses.append(mean_loss)
        val_epochs.append(epoch)
        val_f_score.append(mean_f1)
        time_elapsed.append(time.time()-t0)
        if print:
            print(f"validation loss at epoch {epoch} = {mean_loss}")
            print(f"validation mean f1 score at epoch {epoch} = {mean_f1}")


    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.savefig('GCN_mean_loss.png')
    torch.save(model.state_dict(), "model_state/GCN_WW.pth")

