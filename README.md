# TreeSpeciesClassification
This repostiory contains the code used in the publication [Classification of Tree Species and Standing Dead Trees with Lidar Point Clouds using two Deep Neural Networks: PointCNN and 3DmFV-Net](https://doi.org/10.1007/s41064-022-00200-4).
## Disclaimer
All code was originally written for the master's thesis titled "Tree Species Classification Using Deep Neural Networks on Lidar Point Clouds", which the aforementioned paper is based on. Application of this code to other data was not in mind and a lot of things are hard coded. This was my very first project concerning deep learning. This should just be viewed as research code and nothing that should go into production. I learned a lot from it and in the future there will be more structured code and experiments.

Everything was originally copied to different folders to run the experiments. For some experiments the python files were edited. So this is more a dump of the relevant file, rather than something with a usuable structure. 

## About the networks
Two deep learning networks were used in this thesis: [PointCNN](https://arxiv.org/abs/1801.07791) and [3DmFV-Net](https://doi.org/10.1109/LRA.2018.2850061). 
Both networks are implemented in PyTorch. 

## About the data
The used data input for the networks are segmented trees, which contain each 1,024 3D-points _(x, y, z)_. The point clouds of a tree are all centered around their mean and scaled uniformly to fit into a unit sphere. These points can have more features, like the intensity of the returning laser pulse, the normals, or multispectral information.

The data was given as segmented trees, already preprocessed, in HDF Format. 

## Requirements
Everything was done using [Python](https://www.python.org/), [PyTorch](github.com/pytorch/pytorch), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
- ``Python == 3.6.12``
- ``PyTorch == 1.6.0``
- ``PyTorch Lighning == 1.0.2``
- ``PyTorch Geometric == 1.6.1``
