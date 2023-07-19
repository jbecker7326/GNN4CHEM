from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from train import train
from test import test
def main():
    num_epochs = 10
    lr = 1e-3
    train_dataset = datasets.PPI(root="./data", split="train")
    train_dataset.shuffle()
    val_dataset = datasets.PPI(root="./data", split="val")
    test_dataset = datasets.PPI(root="./data", split="test")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    in_channels = train_dataset.num_features
    out_channels = train_dataset.num_classes

    train(train_loader, num_epochs, in_channels, out_channels, lr, val_loader, debug=True)
    test(test_loader, in_channels, out_channels, num_epochs, lr)
if __name__ == '__main__':
    main()