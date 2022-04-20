import json
import random
import os.path
from typing import *

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, vocab
from tqdm import tqdm

from stud import utils, constants as const
from stud.constants import XML_DATA_SUFFIX, TXT_GOLD_KEYS_SUFFIX, TRANSFORMER_EMBEDDER_SEP_TOKEN
from stud.data import Token
from stud.data_readers import read_wsd_corpus, read_wic_corpus
from stud.sense_inventories import build_senses_vocab, SenseInventory
from stud.transformer_embedder import TransformerEmbedder
from stud.utils import list_to_dict

Sample = Dict[str, Union[Tensor, Token, List[str]]]
Batch = Dict[str, Union[Tensor, List[Token], List[List[str]]]]


class WSDDataset(Dataset):
    """
    Dataset for the token level classification approach to WSD.
    Contextualized embeddings are pre-computed for each sample.
    """

    def __init__(self,
                 samples: List[List[Token]],
                 embedder: str,
                 sense_inventory: SenseInventory,
                 senses_vocab: Optional[Vocab] = None) -> None:
        super().__init__()

        self.senses_vocab = senses_vocab if senses_vocab is not None else build_senses_vocab(samples)

        self.embedder = TransformerEmbedder(embedder)

        self.__encode_samples(samples, sense_inventory)
        self.to_device("cpu")

    @classmethod
    def from_path(cls,
                  path: str,
                  embedder: str,
                  sense_inventory: SenseInventory,
                  senses_vocab: Optional[Vocab] = None) -> "WSDDataset":
        samples = read_wsd_corpus(f"{path}{XML_DATA_SUFFIX}", f"{path}{TXT_GOLD_KEYS_SUFFIX}")
        return cls(samples, embedder, sense_inventory, senses_vocab)

    @classmethod
    def from_preprocessed(cls, file_path: str, device: str = "cpu") -> "WSDDataset":
        return torch.load(file_path, map_location=device)

    @classmethod
    def parse(cls,
              samples_or_path: Union[List[List[Token]], str],
              embedder: str,
              sense_inventory: SenseInventory,
              senses_vocab: Optional[Vocab] = None,
              device: str = "cpu") -> "WSDDataset":

        if isinstance(samples_or_path, str):
            if os.path.isfile(samples_or_path):
                return WSDDataset.from_preprocessed(samples_or_path, device)
            else:
                return WSDDataset.from_path(samples_or_path, embedder, sense_inventory, senses_vocab)
        elif isinstance(samples_or_path, List):
            return cls(samples_or_path, embedder, sense_inventory, senses_vocab)
        else:
            raise Exception("`samples_or_path` is neither a `List[List[Token]]` nor a `str`")

    @property
    def input_size(self) -> int:
        return self.embedder.embedding_dimension

    @property
    def num_senses(self) -> int:
        return len(self.senses_vocab.get_itos())

    def to_device(self, device: str) -> None:
        for sample in self.encoded_samples:
            sample["sense_embedding"] = sample["sense_embedding"].to(device)

    def __encode_samples(self, samples: List[List[Token]], sense_inventory: SenseInventory) -> None:

        self.encoded_samples: List[Sample] = list()

        for sample_idx in tqdm(range(len(samples))):
            sample = samples[sample_idx]

            # shape: (batch_size=1, sample_length, embedding_dimension)
            embeddings: Tensor = self.embedder([[token.text for token in sample]], device=utils.get_device())
            # remove the empty batch size dimension
            embeddings: Tensor = embeddings.squeeze(dim=-1)

            for token, embedding in zip(sample, embeddings):
                if token.is_tagged:

                    # retrieve the possible sense ids of the given token
                    possible_sense_ids = sense_inventory.get_possible_sense_ids(token)

                    self.encoded_samples.append({
                        # store the embedding of the target word's hidden state
                        "sense_embedding": embedding,
                        "sense_index": torch.tensor(self.senses_vocab[token.sense_id]),
                        "candidates": possible_sense_ids,
                        "token": token
                    })

    def __getitem__(self, index: int) -> Sample:
        return self.encoded_samples[index]

    def __len__(self) -> int:
        return len(self.encoded_samples)

    @staticmethod
    def collate_fn(list_batch: List[Sample]) -> Batch:

        dict_batch = list_to_dict(list_batch)

        return {
            "sense_embeddings": torch.stack(dict_batch["sense_embedding"]),
            "sense_indices": torch.stack(dict_batch["sense_index"]),
            "candidates": dict_batch["candidates"],
            "tokens": dict_batch["token"]
        }


class GlossBERTDataset(Dataset):
    """
    Dataset for fine-tuning BERT with the context-gloss pairs binary classification task,
    following: `GlossBERT: BERT for word sense disambiguation with gloss knowledge. <https://aclanthology.org/D19-1355>`
    """

    def __init__(self, samples: List[Sample]) -> None:
        super().__init__()
        self.samples = samples

    @classmethod
    def from_tokens(cls,
                    samples: List[List[Token]],
                    sense_inventory: SenseInventory,
                    senses_vocab: Optional[Vocab] = None) -> "GlossBERTDataset":
        return cls(cls.preprocess(samples, sense_inventory, senses_vocab))

    @classmethod
    def from_json(cls, path: str) -> "GlossBERTDataset":
        with open(path) as f:
            encoded_samples = list()
            for sample in json.load(f):
                encoded_samples.append({
                    "context-gloss": sample["context-gloss"],
                    "label": torch.tensor(sample["label"]),
                    "sense_index": torch.tensor(sample["sense_index"]),
                    "token": Token.parse(sample["token"])
                })
            return cls(encoded_samples)

    @classmethod
    def parse(cls, samples_or_path: Union[List[Sample], str]) -> "GlossBERTDataset":

        if isinstance(samples_or_path, str):
            return cls.from_json(samples_or_path)
        elif isinstance(samples_or_path, List):
            return cls(samples_or_path)
        else:
            raise Exception(f"`samples_or_path` type: {type(samples_or_path)} is not valid")

    def save_as_json(self, path: str, indent: Optional[int] = None) -> None:
        json_samples = list()

        for sample in self:
            json_samples.append({
                "context-gloss": sample["context-gloss"],
                "label": sample["label"].item(),
                "sense_index": sample["sense_index"].item(),
                "token": sample["token"].as_dict()
            })

        with open(path, "w+") as f:
            json.dump(json_samples, f, indent=indent)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def preprocess(samples: List[List[Token]],
                   sense_inventory: SenseInventory,
                   senses_vocab: Optional[Vocab] = None) -> List[Sample]:

        if senses_vocab is None:
            senses_vocab = sense_inventory.build_senses_vocab()

        encoded_samples = list()

        for sample_idx in tqdm(range(len(samples))):
            sample = samples[sample_idx]

            for token in sample:
                if token.is_tagged:
                    # retrieve all possible sense ids and generate context-gloss pairs
                    for sense_id in sense_inventory.get_possible_sense_ids(token):
                        context = [token.text for token in sample]
                        gloss = sense_inventory.get_gloss(sense_id)
                        context_gloss = context + [TRANSFORMER_EMBEDDER_SEP_TOKEN] + gloss

                        # label is True when the context+gloss pair represents the target word's sense, False otherwise
                        label = 1 if sense_id == token.sense_id else 0

                        encoded_samples.append({
                            "context-gloss": context_gloss,
                            "label": torch.tensor(label),
                            "sense_index": torch.tensor(senses_vocab[sense_id]),
                            "token": token
                        })

        return encoded_samples

    @staticmethod
    def collate_fn(list_batch: List[Sample]) -> Batch:

        dict_batch = list_to_dict(list_batch)

        return {
            "context-gloss": dict_batch["context-gloss"],
            "labels": torch.stack(dict_batch["label"]),
            "sense_indices": torch.stack(dict_batch["sense_index"]),
            "tokens": dict_batch["token"]
        }


if __name__ == "__main__":

    training_corpus = read_wsd_corpus(f"{const.TRAIN_SET_PATH}{const.XML_DATA_SUFFIX}",
                                      f"{const.TRAIN_SET_PATH}{const.TXT_GOLD_KEYS_SUFFIX}")
    semeval07_corpus = read_wsd_corpus(f"{const.VALID_SET_PATH}{const.XML_DATA_SUFFIX}",
                                       f"{const.VALID_SET_PATH}{const.TXT_GOLD_KEYS_SUFFIX}")
    evaluation_corpus = read_wsd_corpus(f"{const.TEST_SET_PATH}{const.XML_DATA_SUFFIX}",
                                        f"{const.TEST_SET_PATH}{const.TXT_GOLD_KEYS_SUFFIX}")

    wic_samples_dev = read_wic_corpus(const.WIC_TEST_SET_PATH, const.WIC_TEST_SET_WSD_KEYS_PATH)
    wic_corpus_dev = list()
    for wic_sample in wic_samples_dev:
        wic_corpus_dev.append(wic_sample.sentence1)
        wic_corpus_dev.append(wic_sample.sentence2)

    senses_vocabulary = build_senses_vocab(training_corpus + evaluation_corpus + wic_corpus_dev)
    sense_invent = SenseInventory(const.GLOSSES_PATH, const.LEMMA_POS_DICT_PATH)

    embedder_model = utils.get_pretrained_model(const.TRANSFORMER_EMBEDDER_PATH)

    train_set = WSDDataset(training_corpus, embedder_model, sense_invent, senses_vocabulary)
    torch.save(train_set, const.PREPROCESSED_TRAIN_PATH)

    valid_set = WSDDataset(semeval07_corpus, embedder_model, sense_invent, senses_vocabulary)
    torch.save(valid_set, const.PREPROCESSED_VALID_PATH)

    wic_test_set = WSDDataset(wic_corpus_dev, embedder_model, sense_invent, senses_vocabulary)
    torch.save(wic_test_set, const.PREPROCESSED_TEST_PATH)

    ### GlossBERT datasets
    
    wic_samples_train = read_wic_corpus(const.WIC_TRAIN_SET_PATH)
    wic_corpus_train = list()
    for wic_sample in wic_samples_train:
        wic_corpus_train.append(wic_sample.sentence1)
        wic_corpus_train.append(wic_sample.sentence2)

    SemCor_fraction_size = 0.15
    SemCor_fraction = random.sample(training_corpus, int(len(training_corpus) * SemCor_fraction_size))

    train_set = GlossBERTDataset.from_tokens(SemCor_fraction, sense_invent)
    train_set.save_as_json(f"../../data/preprocessed/SemCor{int(SemCor_fraction_size * 100)}.json")

    valid_set = GlossBERTDataset.from_tokens(semeval07_corpus, sense_invent)    
    valid_set.save_as_json(const.PREPROCESSED_GLOSSBERT_VALID_PATH)

    test_set = GlossBERTDataset.from_tokens(wic_corpus_dev, sense_invent)
    test_set.save_as_json(const.PREPROCESSED_GLOSSBERT_TEST_PATH)
