import os

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from stud import constants, utils
from stud.data import HParams, Token, Pos
from stud.datasets import WSDDataset
from stud.transformer_embedder import TransformerEmbedder
from stud.pl_data_modules import WSDDataModule
from stud.pl_modules import WordSenseDisambiguator

pl.seed_everything(42, workers=True)

utils.nltk_downloads()

data_module = WSDDataModule(constants.PREPROCESSED_TRAIN_PATH,
                            constants.PREPROCESSED_VALID_PATH,
                            constants.PREPROCESSED_TEST_PATH,
                            constants.TRANSFORMER_EMBEDDER_PATH,
                            batch_size=16)

data_module.setup("fit")

train_set = data_module.train_set

hparams = HParams(num_classes=train_set.num_senses,
                  input_size=train_set.input_size)

MODELS_DIR = "../../model/"
MODEL_NAME = "wsd_model_wordnet"

early_stopping = pl.callbacks.EarlyStopping(monitor="valid_wsd_accuracy",
                                            patience=10,
                                            verbose=True,
                                            mode="max")

check_point_callback = pl.callbacks.ModelCheckpoint(monitor="valid_wsd_accuracy",
                                                    verbose=True,
                                                    save_top_k=1,
                                                    save_last=False,
                                                    mode="max",
                                                    dirpath=MODELS_DIR,
                                                    filename=MODEL_NAME + "-{epoch}-{valid_wsd_loss:.4f}-"
                                                                          "{valid_wsd_f1:.3f}-{valid_wsd_accuracy:.3f}")

wandb_logger = WandbLogger(offline=False, project="nlp_hw3", name=MODEL_NAME)

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    logger=wandb_logger,
    val_check_interval=1.0,
    max_epochs=150,
    callbacks=[early_stopping, check_point_callback],
    deterministic=True
)

model = WordSenseDisambiguator(hparams.as_dict(),
                               senses_vocab=train_set.senses_vocab,
                               ignore_loss_index=train_set.senses_vocab[constants.UNK_TOKEN])

trainer.fit(model, datamodule=data_module)

wandb.finish()
