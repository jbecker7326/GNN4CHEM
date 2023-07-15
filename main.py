from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from train import train
from test import test
def main():
    num_epochs = 10
    lr = 1e-3
    train_dataset = datasets.PPI(root="./data", split="train")
    val_dataset = datasets.PPI(root="./data", split="val")
    test_dataset = datasets.PPI(root="./data", split="test")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    for batch in train_loader:
        break
    in_channels = batch.x.shape[1]
    out_channels = train_dataset.num_classes

    train(train_loader, num_epochs, in_channels, out_channels, lr, val_loader)
    test(test_dataset, test_loader, in_channels, out_channels)
if __name__ == '__main__':
    main()