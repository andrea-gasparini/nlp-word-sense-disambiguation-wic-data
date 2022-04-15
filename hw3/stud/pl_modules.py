from dataclasses import asdict
from typing import *

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch import nn
from torchmetrics import Accuracy, F1Score
from torchtext.vocab import Vocab

from stud import utils
from stud.constants import UNK_TOKEN
from stud.datasets import Batch

StepOutput = Dict[str, Tensor]


class WordSenseDisambiguator(pl.LightningModule):

    def __init__(self, hparams: Dict, senses_vocab: Vocab, ignore_loss_index: int = -100) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)
        self.senses_vocab = senses_vocab

        self.model = nn.Sequential(nn.BatchNorm1d(self.hparams.input_size),
                                   nn.Dropout(0.2),
                                   nn.Linear(self.hparams.input_size, self.hparams.hidden_size),
                                   nn.Dropout(0.2),
                                   nn.ReLU(),
                                   nn.Linear(self.hparams.hidden_size, self.hparams.num_classes))

        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_loss_index)

        self.train_acc = Accuracy()
        self.train_f1 = F1Score(average="macro", num_classes=self.hparams.num_classes)
        self.valid_acc = Accuracy()
        self.valid_f1 = F1Score(average="macro", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy()
        self.test_f1 = F1Score(average="macro", num_classes=self.hparams.num_classes)

    def forward(self, batch: Batch) -> StepOutput:
        logits = self.model(batch["sense_embeddings"])
        return {
            "logits": logits,
            "preds": torch.softmax(logits, dim=-1)
        }

    def step(self, batch: Batch) -> StepOutput:
        out = self(batch)
        out["loss"] = self.loss_function(out["logits"], batch["sense_indices"])
        out["predicted_indices"], out["predicted_ids"] = self.predict(batch, out)
        return out

    def training_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        out = self.step(batch)
        self.log_dict({
            f"train_wsd_loss": out["loss"],
            f"train_wsd_accuracy": self.train_acc(out["predicted_indices"], batch["sense_indices"]),
            f"train_wsd_f1": self.train_f1(out["predicted_indices"], batch["sense_indices"]),
        }, prog_bar=True, on_step=False, on_epoch=True)
        return out

    def validation_step(self, batch: Batch, batch_idx: int) -> Optional[StepOutput]:
        out = self.step(batch)
        self.log_dict({
            f"valid_wsd_loss": out["loss"],
            f"valid_wsd_accuracy": self.valid_acc(out["predicted_indices"], batch["sense_indices"]),
            f"valid_wsd_f1": self.valid_f1(out["predicted_indices"], batch["sense_indices"]),
        }, prog_bar=True, on_step=False, on_epoch=True)
        return out

    def test_step(self, batch: Batch, batch_idx: int) -> Optional[StepOutput]:
        out = self.step(batch)
        self.log_dict({
            f"test_wsd_loss": out["loss"],
            f"test_wsd_accuracy": self.test_acc(out["predicted_indices"], batch["sense_indices"]),
            f"test_wsd_f1": self.test_f1(out["predicted_indices"], batch["sense_indices"]),
        }, prog_bar=True, on_step=False, on_epoch=True)
        return out

    def predict(self, batch: Batch, step_output: StepOutput) -> Tuple[Tensor, List[str]]:

        predicted_ids: List[str] = list()
        predicted_indices: List[int] = list()

        for token, preds in zip(batch["tokens"], step_output["preds"]):

            # retrieve from WordNet the possible sense ids of the given token
            possible_sense_ids = utils.get_wn_possible_sense_ids(token)
            # and get only the indices of those senses also available in our vocabulary
            possible_sense_ids_in_vocab = [sense_id for sense_id in possible_sense_ids if sense_id in self.senses_vocab]
            possible_sense_indices = [self.senses_vocab[sense_id] for sense_id in possible_sense_ids_in_vocab]

            # if none of the retrieved senses is in the vocabulary
            if len(possible_sense_ids_in_vocab) == 0:
                # simply take the most frequent sense (i.e. the first one among the ones retrieved from WordNet)
                predicted_sense_id = possible_sense_ids[0]
                predicted_sense_index = self.senses_vocab[UNK_TOKEN]
            else:
                # get the probabilities for each possible sense id
                possible_sense_ids_probabilities = preds[torch.tensor(possible_sense_indices)]
                # and retrieve the most probable sense
                prediction_index = torch.argmax(possible_sense_ids_probabilities)
                predicted_sense_id = possible_sense_ids_in_vocab[prediction_index]
                predicted_sense_index = possible_sense_indices[prediction_index]

            predicted_ids.append(predicted_sense_id)
            predicted_indices.append(predicted_sense_index)

        return torch.tensor(predicted_indices, device=self.device), predicted_ids

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
