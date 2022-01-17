from itertools import product
import numpy as np
import h5py
from tqdm import tqdm

import torch
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import Conv3d, BatchNorm3d, AvgPool3d, ConstantPad3d
from torch.nn import Sequential as Seq



class TreeData(Dataset):
    '''
    This data object loads the data from an HDF file.
    The HDF has to include 'data':
    The normalized tree point clouds [B, N, 3]
        B: Number of trees
        N: Number of points per tree
    And 'label': A prediction label for the tree.
    # This should be handled differently for unlabled data TODO
    Upon loading the data can be augmented (rotation around z-axis)
    and transformed into 3DmFV [default].
    '''

    def __init__(self, datafile, transform=[8, 8, 8], sigma=None,
                 data_augmentation=False, num_rot=3):
        self.transform = transform
        self.sigma = sigma
        self.data_augmentation = data_augmentation
        self.num_rot = num_rot
        with h5py.File(datafile, 'r') as file:
            xyz = file['data'][()]
            xyz = [xyz[i, :, :] for i in range(xyz.shape[0])]
            # Bringing the tree height exactly in the range [-1, 1]
            for i, obj in enumerate(xyz):
                obj[:, -1] -= obj[:, -1].min()
                obj /= obj[:, -1].max()
                obj *= 2
                obj[:, -1] -= 1
                xyz[i] = obj
            labels = file['label'][()]
            self.labels = [labels[i] for i in range(labels.shape[0])]

        # Data Augmentation
        # Rotate the trees around their z-axis
        if data_augmentation:
            rot_angles = np.linspace(0, 2 * np.pi, self.num_rot + 2)[1:-1]
            print('Rotating the trees systematically in {} other positions: {}'.format(self.num_rot,
                                                                                       ', '.join(map(lambda x: '{:.1f}'.format(x) + '°', rot_angles * 180 / np.pi))))
            xyz_rot = []
            for rot in rot_angles:
                rot_mat = np.array(
                    [[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]]).T
                for tree in xyz:
                    xyz_rot.append(tree @ rot_mat)
            print('Adding {} rotated trees to the dataset'.format(len(xyz_rot)))
            xyz = xyz + xyz_rot
            self.labels = self.labels + self.labels * self.num_rot
        else:
            print('No data augmentation')

        self.numtrees = len(xyz)

        # transform the data into 3DmFV
        if self.transform is not None:
            trn = FV3Dm(self.transform, self.sigma)
            print('Transforming point clouds to {} 3DmFV'.
                  format('\u00d7'.join(map(str, transform))))
            self.trees = []
            for tree in tqdm(xyz):
                self.trees.append(trn(tree).reshape((20, *transform)))
            print('Transformation done.')
        else:
            print('No transformation')
            self.trees = xyz

    def __len__(self):
        return self.numtrees

    def __getitem__(self, idx):
        sample = {'points': tensor(self.trees[idx]).float(),
                  'label': tensor(self.labels[idx]).long()}
        return sample

    def __str__(self):
        # if self.transform is not None:
        #     return f'TreeData(numtrees={self.numtrees}, transform=True, grid={self.transform})'
        # else:
        #     return f'TreeData(numtrees={self.numtrees}, transform=False)'
        return 'TreeData(numtrees={}, transform={}{}, data_augmentation={}{})'.\
            format(self.numtrees,
                   self.transform is not None,
                   ', grid={}'.format(self.transform)
                   if self.transform is not None else '',
                   self.data_augmentation,
                   ', rotations={}'.format(self.num_rot)
                   if self.transform else '')

    def __repr__(self):
        return self.__str__()


class FV3Dm(object):
    '''
    Transformer object to transform the point clouds into 3DmFVs
    '''

    def __init__(self, grid=[8, 8, 8], std=None):
        self.means, self.std, self.weights = get_3D_grid(grid, std)
        self.grid = grid

    def __call__(self, points):
        points = get_3DmFV(points, self.means, self.std, self.weights)
        return points


class CMDisplay:
    def __init__(self, confusion_matrix, *, display_labels=['coniferous',
                                                            'decidious',
                                                            'snag',
                                                            'dead tree']):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, *, include_values=True, cmap='Blues',
             xticks_rotation='vertical', values_format=None,
             ax=None, colorbar=False):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


class Inception(torch.nn.Module):
    """
    Implementation of the Inception module, described in the paper and adopted from the original code.
    This is a PyTorch implementation of the original code.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 64,
        kernel_c1: int = 4,
        kernel_c2: int = 8,
        activation_fn=F.relu,
    ):

        super(Inception, self).__init__()

        self.n_filters = n_filters
        self.kernel_c1 = kernel_c1
        self.kernel_c2 = kernel_c2
        self.in_channels = in_channels
        self.activation_fn = activation_fn
        self.pad_c1 = (int(kernel_c1 / 2), int((kernel_c1 - 1) / 2)) * 3
        self.pad_c2 = (int(kernel_c2 / 2), int((kernel_c2 - 1) / 2)) * 3

        self.cnn0 = Seq(
            Conv3d(in_channels=in_channels,
                   out_channels=n_filters, kernel_size=1),
            BatchNorm3d(n_filters),
        )
        self.avgpool = Seq(
            ConstantPad3d(self.pad_c1, 0), AvgPool3d(
                kernel_size=kernel_c1, stride=1)
        )
        self.cnn1 = Seq(
            Conv3d(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
            BatchNorm3d(n_filters),
        )
        self.cnn2 = Seq(
            ConstantPad3d(self.pad_c1, 0),
            Conv3d(
                in_channels=n_filters,
                out_channels=n_filters // 2,
                kernel_size=kernel_c1,
            ),
            BatchNorm3d(n_filters // 2),
        )
        self.cnn3 = Seq(
            ConstantPad3d(self.pad_c2, 0),
            Conv3d(
                in_channels=n_filters,
                out_channels=n_filters // 2,
                kernel_size=kernel_c2,
            ),
            BatchNorm3d(n_filters // 2),
        )

    def forward(self, x):
        """
        The concatenation is in the same order as the original
        """
        x0 = self.activation_fn(self.cnn0(x))
        x1 = self.activation_fn(self.cnn2(x0))
        x2 = self.activation_fn(self.cnn3(x0))
        x3 = self.activation_fn(self.cnn1(self.avgpool(x0)))
        return torch.cat((x0, x1, x2, x3), dim=1)


def get_3D_grid(subdivisions=[8, 8, 8], std=None, astensor=False):
    '''
    This function is returning the positions (means), standard deviations and
    weights of a 3D GMM. The Gaussians lay on a uniform grid. Spherical, or in
    an 'un-cubic' case ellipsoidic, Gaussians are assumed. The weights are
    also set to be the same. The result can be returned as tensors.
    '''

    # If no standard deviation is given (default) they are calculated.
    if std is None:
        std = np.array([1 / x for x in subdivisions])
    divisions = np.array(subdivisions)
    n_gaussians = divisions.prod()
    # This is to be flexible. So despite the function name also nD grids can be created
    D = divisions.shape[0]
    means = np.r_[
        '-1', np.meshgrid(*([np.linspace(-1, 1, d) for d in divisions]))].reshape(D, -1).T
    std = std * np.ones_like(means)
    w = (1. / n_gaussians) * np.ones((n_gaussians, 1))
    if astensor:
        return (tensor(means).float(), tensor(std).float(), tensor(w).float().view(-1))
    # to be compliant with the 3DmFV numpy function below the weights are a 1D array
    return (means, std, w[:, 0])


def get_3DmFV(tree, mu, sigma, w=None):
    '''
    Input:
        tree [N, D]: tree lidar points
        mu [K, D]: Centers of the Gaussians
        sigma [K, D]: Sigmas of the Gaussian (spherical)
        w [K]: Weights of the Gaussians

    N: the number of points in a tree
    D: the dimension of the space (XYZ = 3)
    K: the number of gaussians

    Output:
        fv [K, (D * 9 + 2)]: 3D-modified Fisher Vectors
    '''

    # Enforce sum of weights to be 1
    if w is None:
        w = np.ones(mu.shape[0])
    if w.sum() != 1:
        w = np.exp(w) / np.exp(w).sum()

    # get dimensional values
    N, D = tree.shape
    K, _ = mu.shape

    # expand the tree to be ready for calculation [N, K, D]
    tree = np.repeat(tree[:, np.newaxis, :], K, axis=1)
    # or: np.transpose(np.tile(tree, (K, 1, 1)), (1, 0, 2))

    # Calculate the probability for each point belonging to the Gaussians
    prob_tree = tree - mu
    prob_tree = np.exp(
        np.sum(-0.5 * prob_tree ** 2 * sigma, axis=-1))  # [N, K]
    prob_tree = prob_tree / ((np.pi * 2) ** (D / 2) * sigma.prod(axis=1))
    prob_tree = prob_tree * w
    prob_tree = (prob_tree.T / prob_tree.sum(axis=1)).T  # [N, K]

    d_alpha = (prob_tree - w) / (np.sqrt(w) * N)
    d_alpha = np.stack(
        (d_alpha.max(axis=0), d_alpha.sum(axis=0)), axis=1)  # [K, 2]

    d_mu = prob_tree[:, :, np.newaxis] * \
        ((tree - mu) / sigma) / (np.sqrt(w[:, np.newaxis]) * N)  # [N, K, D]
    d_mu = np.concatenate((d_mu.max(axis=0), d_mu.min(
        axis=0), d_mu.sum(axis=0)), axis=1)  # [K, D*3]

    d_sigma = prob_tree[:, :, np.newaxis] * (((tree - mu) / sigma)**2 - 1) / (
        np.sqrt(w[:, np.newaxis] * 2) * N)  # [N, K, D]
    d_sigma = np.concatenate((d_sigma.max(axis=0), d_sigma.min(
        axis=0), d_sigma.sum(axis=0)), axis=1)  # [K, D*3]

    fv = np.concatenate((d_alpha, d_mu, d_sigma), axis=1)  # [K, D*9 + 2]

    # Power normalization
    alpha = 0.5
    epsilon = 1e-12

    fv = np.sign(fv) * (np.maximum(np.abs(fv), epsilon) ** alpha)

    # l2-normalization
    fv = fv / np.linalg.norm(fv, axis=0)

    return fv.T  # [D*9 + 2, K]

