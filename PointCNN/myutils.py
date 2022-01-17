from itertools import product
import h5py
from torch_geometric.data import Data, DataLoader
from torch import tensor
import numpy as np


def load_dataset(path, batch_size=16, shuffle=False, load_intens=True, load_ms=False):
    """Generates a torch_geometric Dataloader from an HDF-File at the specified path.
    The batch size of the dataloader can be specified.

    Args:
        path (str): Path to HDF (.h5) file
        batch_size (int, optional): Batch size of the resulting data loader object. Defaults to 16.

    Returns:
        DataLoader object
    """
    dataset = []
    with h5py.File(path, 'r') as file:
        for idx in range(file['id'].shape[0]):
            dataset.append(
                Data(
                    pos=tensor(file['data'][idx]),
                    y=tensor(file['label'][idx]).long(),
                    x=tensor(file['intens'][idx]) if load_intens else None,
                    feat=tensor([file['feature'][idx][0]]) if load_ms else None
                ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_dataset_scaleintens(path, batch_size=16, shuffle=False, load_ms=False):
    """Generates a torch_geometric Dataloader from an HDF-File at the specified path.
    The batch size of the dataloader can be specified.

    Args:
        path (str): Path to HDF (.h5) file
        batch_size (int, optional): Batch size of the resulting data loader object. Defaults to 16.

    Returns:
        DataLoader object
    """
    dataset = []
    with h5py.File(path, 'r') as file:
        intens = file['intens'][()]
        # some magic numbers. 4.3 is approx. the mean of the whole dataset and around 1. So 3 sigma fits into the range [-1, 1]
        intens = (np.log(intens + 1) - 4.3) / 3
        for idx in range(file['id'].shape[0]):
            dataset.append(
                Data(
                    pos=tensor(file['data'][idx]),
                    y=tensor(file['label'][idx]).long(),
                    x=tensor(intens[idx]),
                    feat=tensor([file['feature'][idx][0]]) if load_ms else None
                ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_dataset_special(path, batch_size=16, shuffle=False):
    """Generates a torch_geometric Dataloader from an HDF-File at the specified path.
    The batch size of the dataloader can be specified.

    Args:
        path (str): Path to HDF (.h5) file
        batch_size (int, optional): Batch size of the resulting data loader object. Defaults to 16.

    Returns:
        DataLoader object
    """
    dataset = []
    with h5py.File(path, 'r') as file:
        intens = file['intens'][()]
        ms = file['feature'][()]
        features = np.concatenate((np.atleast_3d(intens), ms), axis=2)
        for idx in range(file['id'].shape[0]):
            dataset.append(
                Data(
                    pos=tensor(file['data'][idx]),
                    y=tensor(file['label'][idx]).long(),
                    x=tensor(features[idx])
                ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_dataset_scaleintens_special(path, batch_size=16, shuffle=False):
    """Generates a torch_geometric Dataloader from an HDF-File at the specified path.
    The batch size of the dataloader can be specified.

    Args:
        path (str): Path to HDF (.h5) file
        batch_size (int, optional): Batch size of the resulting data loader object. Defaults to 16.

    Returns:
        DataLoader object
    """
    dataset = []
    with h5py.File(path, 'r') as file:
        intens = file['intens'][()]
        # some magic numbers. 4.3 is approx. the mean of the whole dataset and around 1. So 3 sigma fits into the range [-1, 1]
        intens = (np.log(intens + 1) - 4.3) / 3
        ms = file['feature'][()]
        features = np.concatenate((np.atleast_3d(intens), ms), axis=2)
        for idx in range(file['id'].shape[0]):
            dataset.append(
                Data(
                    pos=tensor(file['data'][idx]),
                    y=tensor(file['label'][idx]).long(),
                    x=tensor(features[idx])
                ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# This is just a copy of the sklearn.metrics Class as the official version gives an error


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
