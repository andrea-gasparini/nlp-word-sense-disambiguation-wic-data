import os
from typing import *

import torch
from torch import Tensor
from torch.nn.utils import rnn

from stud.constants import PAD_INDEX

T = TypeVar('T')


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
