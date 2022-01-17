# PointCNN
This is the PyTorch implementation of [PointCNN](https://arxiv.org/abs/1801.07791). It uses the [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) implementation o the $\mathcal{X}$-Conv layers. The model structure is modeled after the structure for the ModelNet40 classification in the paper.

The files contain different experiments:

- `pointcnn_geom` is just using the geometry of the point clouds, i.e. $x$,$y$ and $z$
- `pointcnn_geom+i` is using the geometry and the laser intensities as features
- `pointcnn_geom+i+ms` like the above and using 5 additional multi spectral features of the whole tree in the FCN at the end of the network
- `pointcnn_geom+i+ms_notlin` is using the MS features as additional features. So each point has the same MS features
- `pointcnn_geom+i_scaled` the intensities are scaled
- `pointcnn_geom+i_scaled+ms` scaled intensities and MS features in the FCN
- `pointcnn_geom+i_scaled+ms_notlin` scaled intensities an MS featues as additional point features
- Everything containing `+n` uses the normals