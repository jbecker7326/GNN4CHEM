# GNN4CHEM

Team: **GNN4CHEM**

Project title: **Node Classification on Protein-Protein Interactions for Applications in Biochemistry**

## Project Abstract

The purpose of this project is to explore and advance Graph Neural Network (GNN) architectures to make multi-label predictions on the nodes of Protein-Protein Interactions (PPIs). By testing existing GNN layer types in different architectures and comparing against benchmarks, our study found that a combination of Molecular Fingerprint Convolution (MFConv) and Graph Isomorphism Network (GIN) layers produced our best F-1 scores across validation and test data. This impressive result is analyzed from the theoretical perspective on both architectures in terms of discriminating power.

## Code Use & Installation

1. Download the repository from the GitHub link: https://github.gatech.edu/wwang674/GNN4CHEM

2. To set up the environment, run:

 ```conda env create -f environment.yml```

3. Install PyTorch for you machine using the pip command specified at the following link:
https://pytorch.org/get-started/locally/

### Quick-Start Guide: Recreating Data for Report

To generate the plots and csv used in gnn4chem.PDF without training the model, run:
`python main.py -conn gnn4chem.db -read_db True`

To generate the plots and csv used in gnn4chem.PDF WITH training the model (**NOTE This may take >1 day even on a CUDA-enabled machine**):
`python main.py`

Both commands will generate the following files:
- The data from table 2 will be stored in `'train_stats_ne8000_lr0.001_nj100.csv'` and `'test_stats_ne8000_lr0.001_nj100.csv'`
- Figure 2 will be stored in `'combined_learning_curve_ne8000_lr0.001_nj100.png'`

### Full Guide

Our project has the following arguments that can be specified when running `main.py` from the command line to control the sampling, training, hyper parameter tuning, and SQL-saving for the model:

* `-ne/--num_epochs`: Number of epochs for training. (Type = int, default = 8000)
* `-lr/--learning_rate`: Learning rate for optimizer. (Type = float, default = 1e-3)
* `-ns/--num_smooth`: Smoothing factor for the plots. Recommended to visualize distinctions between learning curves of different models with a large number of epochs. (Type = int, default = 1)
* `-nj/--num_jobs`: Number of jobs for cross validation. (Type = int, default = 5)
* `-bs/--batch_size`: Batch size for minibatch specification when initializing training data loader. (Type = int, default = 8)
* `-conn/--sql_connection`: Path for sqlite database connection. Will create and/or save to database if one is specified. (Type = str, default = None)
* `-read_db/--read_db`: Trigger to read from sql database instead of running models. Will not run new models if set to true. Will not work if -conn is not specified. (Type = bool, default = False)

The following example call will generate the data tables and figures from gnn4chem.PDF:

```python main.py -ne 8000 -lr 1e-3 -ns 1 -nj 5 -bs 8 -conn gnn4chem.db -read_db True```


Running `main.py` will generate the following files 1-4, where the arguments in curly brackets {} will be replaced by those specified in the call to main.py. If these arguments are unused, the files will save with the default values specified above.

1. Training statistics are averaged over `-nj` runs and batch size `-b`. File name `train_stats_ne{-ne}_lr{-lr}_nj{-nj}.csv`. Includes the following:
	- train_loss: Final model training loss
	- train_mean_f1: Final model training f1 score
	- val_loss: Final model validation loss
	- val_mean_f1: Final model validation f1 score
	- time: Model training runtime in seconds
	- mem_alloc: Cuda memory allocated to model on GPU
	- mem_res: Cuda memory reserved/remaining on GPU

2. Multi-label test statistics generated between the ground-truth labels and predicted labels. File name `test_stats_ne{-ne}_lr{-lr}_nj{-nj}.csv`. Includes the following:
	- MSE: Mean-squared error
	- accuracy: Accuracy
	- precision: Precision
	- sensitivity: Sensitivity/true positive rate
	- specificity: Specificity/true negative rate/recall
	- f1: F1-score. This is our main performance metric to account for imbalance in the PPI dataset.
	- MCC: Matthews Correlation Coefficient (**NOTE: Due to a bug in pytorch-geometric, this performance metric cannot be generated without altering the package. Currently commented out in the code for test.py**)
	- AUROC: Area under the receiver operating curve.
	- AUPRC: Area under the precision recall curve.

3. One learning curve for each model including the validation (navy) and training (orange) curves. Saved to images folder with file names `{model_name}_learning_curve_ne{-ne}_lr{-lr}_nj{-nj}.png`.

4. Combined learning curves for all models in a two-plot figure with training on the left and validation on the right. Saved to images folder with file name `combined_learning_curve_ne{-ne}_lr{-lr}_nj{-nj}.png`
