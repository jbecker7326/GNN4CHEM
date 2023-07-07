from torch_geometric.loader import DataLoader
from models import GNN_my_model
from train import train
from test import test
def main():
    num_epochs = 10
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

    # train(train_loader, num_epochs, in_channels, out_channels, lr)
    test(test_dataset, test_loader, in_channels, out_channels)
if __name__ == '__main__':
    main()