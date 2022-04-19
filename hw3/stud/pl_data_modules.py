from typing import *

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from stud.data import Token
from stud.datasets import WSDDataset
from stud.sense_inventories import SenseInventory


class WSDDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_samples_or_path: Union[List[List[Token]], str],
                 valid_samples_or_path: Union[List[List[Token]], str],
                 test_samples_or_path: Union[List[List[Token]], str],
                 embedder_model: str,
                 sense_inventory: SenseInventory,
                 senses_vocab: Optional[Vocab] = None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = False) -> None:
        """
        Args:
            train_samples_or_path: either a list of training samples or a path to an already preprocessed dump
            valid_samples_or_path: either a list of validation samples or a path to an already preprocessed dump
            test_samples_or_path: either a list of test samples or a path to an already preprocessed dump
            senses_vocab: the mapping vocabulary from sense keys to numeric indices
            batch_size: how many samples per batch to load
            num_workers: how many subprocesses to use for data loading
            pin_memory: if ``True``, Tensors are copied into CUDA pinned memory before returning them
        """
        super().__init__()

        self.train_samples_or_path = train_samples_or_path
        self.valid_samples_or_path = valid_samples_or_path
        self.test_samples_or_path = test_samples_or_path

        self.train_set: Optional[WSDDataset] = None
        self.valid_set: Optional[WSDDataset] = None
        self.test_set: Optional[WSDDataset] = None

        self.embedder_model = embedder_model
        self.sense_inventory = sense_inventory
        self.senses_vocab = senses_vocab
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:
            self.train_set = WSDDataset.parse(self.train_samples_or_path,
                                              self.embedder_model,
                                              self.sense_inventory,
                                              self.senses_vocab)
            self.valid_set = WSDDataset.parse(self.valid_samples_or_path,
                                              self.embedder_model,
                                              self.sense_inventory,
                                              self.senses_vocab)

        if stage == "test" or stage is None:
            self.test_set = WSDDataset.parse(self.test_samples_or_path,
                                             self.embedder_model,
                                             self.sense_inventory,
                                             self.senses_vocab)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=self.batch_size,
                          collate_fn=WSDDataset.collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_set,
                          shuffle=False,
                          batch_size=self.batch_size,
                          collate_fn=WSDDataset.collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_set,
                          shuffle=False,
                          batch_size=self.batch_size,
                          collate_fn=WSDDataset.collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
