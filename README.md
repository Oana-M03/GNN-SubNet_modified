![GNNSubNetLogo](https://github.com/pievos101/GNN-SubNet/blob/main/GNNSubNet_plot.png)

# GNN-SubNet: Disease Subnetwork Detection with Explainable Graph Neural Networks

**Warning**: This is a development branch and installatio via pip is work in progress!

## https://www.biorxiv.org/content/10.1101/2022.01.12.475995v1

## Installation

To install GNNSubNet run:

```python
pip install GNNSubNet
```

## Usage

First import GNNSubNet and create a GNNSubNet object:

```python
import GNNSubNet as gnn

# SYNTHETIC ------------------------- #
LOC   = "/home/bastian/GNNSubNet-Project/SYNTHETIC"
PPI   = f'{LOC}/NETWORK_synthetic.txt'
FEATS = [f'{LOC}/FEATURES_synthetic.txt']
TARG  = f'{LOC}/TARGET_synthetic.txt'

g = gnn.GNNSubNet(LOC, PPI, FEATS, TARG)

g.train()

g.gene_names
g.accuracy
g.confusion_matrix

g.explain(4)

g.edge_mask
g.modules
g.modules[0]

g.module_importances

```

The main file is called 'OMICS_workflow.py'.
Within that python file you find the function 'load_OMICS_dataset()'. 
It expects the PPI network as an input, the feature matrices, as well as the outcome class. The input needs to be adjusted by the user.

The PPI network consists of three columns.
The first two columns reflect the edges between the nodes (gene names), the third column is the confidence score of the specified edge. The range of this score is [0,999], where high scores mean high confidence.

The rows of the feature matrices (e.g mRNA, and DNA Methylation) reflect the patients, whereas the columns represent the features (genes). Row names as well as column names are required!

Please see the folder "datasets/TCGA" for some sample/example files.

To execute the script simply type 'python OMICS_workflow.py' within your console.

The mentioned OMICS workflow performs GNN classification, explanations, and community detection for disease subnetwork discovery. 

After exectution of 'OMICS_workflow.py', importance scores are stored in the 'edge_mask.txt' file of the data folder. 

The detected disease subnetworks can be found within the 'communities.txt' file, and the corresponding scores within the 'communities_scores.txt' file.

Note, the mentioned community file contain the network node identifier which match the order of the gene names stored in 'gene_names.txt'

  
