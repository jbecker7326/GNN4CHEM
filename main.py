import time
import torch
from models import GNN_my_model
from torch_geometric.loader import DataLoader

def main():
    num_epochs = 10000
    lr = 1e-3
    dataset = GNN_my_model.PPI(root="./data")
    dataset = dataset.shuffle()
    test_dataset = dataset[:2]
    train_dataset = dataset[2:]
    test_loader = DataLoader(test_dataset, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    for batch in train_loader:
        break
    in_channels = batch.x.shape[1]
    out_channels = dataset.num_classes
    model = GNN_my_model.GCNConvNet(in_channels=in_channels, out_channels=out_channels)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(my_device)

    losses = []
    time_elapsed = []
    epochs = []
    t0 = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch.to(my_device))
            loss = loss_fn(pred, batch.y.to(my_device))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            batch_count += 1
        mean_loss = total_loss / batch_count
        losses.append(mean_loss)
        epochs.append(epoch)
        time_elapsed.append(time.time()-t0)
        if epoch % 100 == 0:
            print(f"loss at epoch {epoch} = {mean_loss}")

if __name__ == '__main__':
    main()
