# GNN4CHEM

Team: **GNN4CHEM**

Project title: **Node Classification on Protein-Protein Interactions for Applications in Biochemistry**

## Project Summary 

Protein-protein interactions (PPIs) are essential for many biological processes, including cell signaling, metabolism, and gene regulation Understanding these interactions are crucial for developing new drugs and treatments. After reviewing the paper [1], which analyzed the performance of simple GNNS, GINs, GCNs and GraphSage on multiple datasets, we found our motivation to design our own one and test for efficacy on the PPI dataset. Additionally, two of our group members have a background in chemistry and are interested in relating the concepts of organic chemistry to various GNN classification problems. Specifically, node classification has many organic chemistry applications -- such as understanding the connections between molecules interacting with one another or eliciting a response, as well as how and when they interact.  

## Approach 

Our approach to this project will be to first implement 3-4 different GNN frameworks for node classification (such as spectral-based approaches, spatial-based approaches and GraphSage as explored in [1]) on the Protein-Protein Interactions (PPI) dataset, which was first introduced in [3], and happens to be part of the PyTorch Geometric and Deep Graph Library (DGL) packages. We will then perform a preliminary analysis to compare the learning and performance of each GNN framework, to aid in designing our own custom GNN.
We will compare the performance (F1 score) of our designed GNN against existing GNN results that have been published. Though we are aware that our networks might not yield strong results since we're implementing some new frameworks and analyses for the dataset, but even these negative results could be helpful for future work in GNN classification. 

To set up the environmnet, run:

 ```conda env create -f environment.yml```
