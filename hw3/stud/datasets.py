import os.path
from typing import *
from xml.etree import ElementTree

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, vocab
from tqdm import tqdm

from stud import utils, constants as const
from stud.constants import XML_DATA_SUFFIX, TXT_GOLD_KEYS_SUFFIX, UNK_TOKEN
from stud.data import Token, Pos
from stud.transformer_embedder import TransformerEmbedder
from stud.utils import list_to_dict

Sample = Dict[str, Union[Tensor, Token]]
Batch = Dict[str, Union[Tensor, List[Token]]]


class WSDDataset(Dataset):

    def __init__(self, samples: List[List[Token]], embedder: str, senses_vocab: Optional[Vocab] = None) -> None:
        super().__init__()

        self.raw_samples = samples
        self.senses_vocab = senses_vocab if senses_vocab is not None else build_senses_vocab(samples)

        self.embedder = TransformerEmbedder(embedder, device=utils.get_device())

        self.__encode_samples()
        self.to_device("cpu")

    @classmethod
    def from_path(cls, path: str, embedder: str, senses_vocab: Optional[Vocab] = None) -> "WSDDataset":
        samples = read_wsd_corpus(f"{path}{XML_DATA_SUFFIX}", f"{path}{TXT_GOLD_KEYS_SUFFIX}")
        return cls(samples, embedder, senses_vocab)

    @classmethod
    def from_preprocessed(cls, file_path: str, device: str = "cpu") -> "WSDDataset":
        return torch.load(file_path, map_location=device)

    @classmethod
    def parse(cls,
              samples_or_path: Union[List[List[Token]], str],
              embedder: str,
              senses_vocab: Optional[Vocab] = None,
              device: str = "cpu") -> "WSDDataset":

        if isinstance(samples_or_path, str):
            path = samples_or_path
            if os.path.isdir(path):
                return WSDDataset.from_path(path, embedder, senses_vocab)
            elif os.path.isfile(path):
                return WSDDataset.from_preprocessed(path, device)
            else:
                raise Exception(f"{path} is not a valid path to a WSD dataset (neither vanilla nor preprocessed)")
        elif isinstance(samples_or_path, List):
            return cls(samples_or_path, embedder, senses_vocab)
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

    def __encode_samples(self) -> None:

        self.encoded_samples: List[Sample] = list()

        for sample_idx in tqdm(range(len(self.raw_samples))):
            sample = self.raw_samples[sample_idx]

            # shape: (batch_size=1, sample_length, embedding_dimension)
            embeddings: Tensor = self.embedder([[token.text for token in sample]])
            # remove the empty batch size dimension
            embeddings: Tensor = embeddings.squeeze()

            for idx, (token, embedding) in enumerate(zip(sample, embeddings)):
                if token.is_tagged:
                    self.encoded_samples.append({
                        "sense_embedding": embedding,
                        "sense_index": torch.tensor(self.senses_vocab[token.sense_id]),
                        "token": token
                    })

    def __getitem__(self, index: int) -> Sample:
        return self.encoded_samples[index]

    def __len__(self) -> int:
        return len(self.encoded_samples)

    @staticmethod
    def collate_fn(batch: List[Sample]) -> Batch:

        dict_batch = list_to_dict(batch)

        return {
            "sense_embeddings": torch.stack(dict_batch["sense_embedding"]),
            "sense_indices": torch.stack(dict_batch["sense_index"]),
            "tokens": dict_batch["token"]
        }


def read_wsd_corpus(xml_data_path: str, txt_gold_keys_path: str) -> List[List[Token]]:

    if not os.path.isfile(xml_data_path):
        raise Exception(f"{xml_data_path} is not a valid xml data file")
    if not os.path.isfile(txt_gold_keys_path):
        raise Exception(f"{txt_gold_keys_path} is not a valid txt gold keys file")

    sentences = list()

    with open(txt_gold_keys_path) as f:
        gold_keys = [line.strip().split(' ') for line in f]

    sense_ids_dict = dict()
    for gold_key in gold_keys:
        token_id = gold_key[0]
        sense_id = gold_key[1]  # ignore eventual secondary senses ([2:])
        sense_ids_dict[token_id] = sense_id

    # iterate over <sentence> tags from the given xml file
    for i, sent_xml in enumerate(ElementTree.parse(xml_data_path).iter('sentence')):

        sentence = list()
        # for each inner xml token (either <instance> or <wf>)
        for token_index, token_xml in enumerate(sent_xml):
            # store the token's sense id if it is <instance>,
            sense_id = (sense_ids_dict.get(token_xml.attrib.get('id'), None)
                        if token_xml.tag == 'instance' else None)

            token = Token(text=token_xml.text.lower(),
                          index=token_index,
                          lemma=token_xml.attrib.get('lemma'),
                          pos=Pos.parse(token_xml.attrib.get('pos')),
                          id=token_xml.attrib.get('id'),
                          sense_id=sense_id)

            sentence.append(token)

        # check that at least one token in the sentence is correctly tagged with a sense id
        if any([token.is_tagged for token in sentence]):
            sentences.append(sentence)

    return sentences


def build_senses_vocab(samples: List[List[Token]]) -> Vocab:
    counter = Counter()

    for sample in samples:
        for token in sample:
            if token.is_tagged:
                counter[token.sense_id] += 1

    vocabulary = vocab(counter, min_freq=1, specials=[UNK_TOKEN])
    vocabulary.set_default_index(vocabulary[UNK_TOKEN])

    return vocabulary


if __name__ == "__main__":

    preprocessed_paths = [const.PREPROCESSED_TRAIN_PATH, const.PREPROCESSED_VALID_PATH, const.PREPROCESSED_TEST_PATH]

    if all([os.path.isfile(prepro_set) for prepro_set in preprocessed_paths]):
        print(f"All the datasets have been already preprocessed and saved in: \n"
              f"\t - training: {const.PREPROCESSED_TRAIN_PATH} \n"
              f"\t - validation: {const.PREPROCESSED_VALID_PATH} \n"
              f"\t - test: {const.PREPROCESSED_TEST_PATH} \n")
        exit(0)

    training_corpus = read_wsd_corpus(f"{const.TRAIN_SET_PATH}{const.XML_DATA_SUFFIX}",
                                      f"{const.TRAIN_SET_PATH}{const.TXT_GOLD_KEYS_SUFFIX}")
    evaluation_corpus = read_wsd_corpus(f"{const.TEST_SET_PATH}{const.XML_DATA_SUFFIX}",
                                        f"{const.TEST_SET_PATH}{const.TXT_GOLD_KEYS_SUFFIX}")

    senses_vocabulary = build_senses_vocab(training_corpus + evaluation_corpus)

    embedder_model = utils.get_pretrained_model(const.TRANSFORMER_EMBEDDER_PATH)

    train_set = WSDDataset(training_corpus, embedder_model, senses_vocabulary)
    valid_set = WSDDataset.from_path(const.VALID_SET_PATH, embedder_model, senses_vocabulary)
    test_set = WSDDataset(evaluation_corpus, embedder_model, senses_vocabulary)

    for dataset, preprocessed_path in [(train_set, const.PREPROCESSED_TRAIN_PATH),
                                       (valid_set, const.PREPROCESSED_VALID_PATH),
                                       (test_set, const.PREPROCESSED_TEST_PATH)]:
        torch.save(dataset, preprocessed_path)
