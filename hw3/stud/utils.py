import os
from typing import *

import nltk
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.nn.utils import rnn

from stud.constants import PAD_INDEX

T = TypeVar('T')


def nltk_downloads(download_dir: Optional[str] = None) -> None:
    """
    Downloads NLTK's necessary resources
    """
    nltk.download('wordnet', download_dir=download_dir)


def get_pretrained_model(pretrained_model_name_or_path: str) -> str:
    """
    Returns the model name or the path to its local cached directory in case the given arg is a valid one
    """
    return (pretrained_model_name_or_path
            if os.path.exists(pretrained_model_name_or_path)
            else os.path.basename(os.path.normpath(pretrained_model_name_or_path)))


def flatten(list_of_lists: List[List[T]]) -> List[T]:
    return [element for inner_list in list_of_lists for element in inner_list]


def list_to_dict(list_of_dictionaries: List[Dict[str, T]]) -> Dict[str, List[T]]:
    return {key: [dictionary[key] for dictionary in list_of_dictionaries] for key in list_of_dictionaries[0]}


def pad_sequence(sequences: Union[List[Tensor], Tensor]) -> Tensor:
    return rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_INDEX)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def plot_confusion_matrix(gold_labels, predictions, normalize=False, title=None, color_map=plt.cm.Blues):
    """
    Prints and plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    classes = np.array([True, False])

    cm = confusion_matrix(gold_labels, predictions, labels=classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=color_map)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.25
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

    return plt, ax
