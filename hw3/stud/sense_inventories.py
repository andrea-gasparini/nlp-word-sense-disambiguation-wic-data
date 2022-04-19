import json
from abc import ABC as ABSTRACT_CLASS, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Tuple

from nltk import TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.corpus.reader import Lemma
from torchtext.vocab import Vocab, vocab
from tqdm import tqdm

from stud import constants
from stud.constants import UNK_TOKEN, XML_DATA_SUFFIX, TXT_GOLD_KEYS_SUFFIX
from stud.data import Token, Pos
from stud.data_readers import read_wsd_corpus, read_wic_corpus
from stud.utils import flatten


def build_senses_vocab(samples: List[List[Token]]) -> Vocab:
    counter = Counter()

    for sample in samples:
        for token in sample:
            if token.is_tagged:
                counter[token.sense_id] += 1

    vocabulary = vocab(counter, min_freq=1, specials=[UNK_TOKEN])
    vocabulary.set_default_index(vocabulary[UNK_TOKEN])

    return vocabulary


class SenseInventory:

    def __init__(self, glosses_path: str, sense_ids_dict_path: str) -> None:
        super().__init__()

        self.tokenizer = TreebankWordTokenizer()

        glosses = read_wsd_corpus(f"{glosses_path}{XML_DATA_SUFFIX}", f"{glosses_path}{TXT_GOLD_KEYS_SUFFIX}")
        self.sense_id_to_gloss: Dict[str, List[Token]] = {gloss[0].sense_id: gloss for gloss in glosses}

        with open(sense_ids_dict_path) as file:
            self.lemma_pos_to_sense_ids: Dict[str, Dict[str, List[str]]] = json.load(file)

    def get_possible_sense_ids(self, token: Token) -> List[str]:
        """
        Returns all possible sense ids matching lemma and part of speech tag (any if pos is None) of the given token
        """
        if token.lemma in self.lemma_pos_to_sense_ids:
            if token.pos is None:
                return flatten(list(self.lemma_pos_to_sense_ids[token.lemma].values()))
            else:
                return self.lemma_pos_to_sense_ids[token.lemma][str(token.pos)]
        else:
            # retrieve sense ids from wordnet API
            return self.get_wn_possible_sense_ids(token.lemma, token.wn_pos)

    def get_gloss(self, sense_id: str) -> List[str]:
        """
        Returns the gloss (definition) associated to the sense id of the given token
        """

        if sense_id in self.sense_id_to_gloss:
            gloss = self.sense_id_to_gloss[sense_id]
            gloss = [token.text for token in gloss]
        else:
            # retrieve gloss from wordnet API
            gloss, lemma = self.__get_wn_gloss(sense_id)
            # add a weak supervision on the gloss by adding the target word's lemma as prefix
            gloss = f"{lemma}: {gloss}"
            gloss = self.tokenizer.tokenize(gloss)

        return gloss

    def build_senses_vocab(self) -> Vocab:
        """
        Returns a `Vocab` instance defining the mapping from the sense ids to a numerical value
        """
        counter = Counter()

        for lemma in self.lemma_pos_to_sense_ids.keys():
            for sense_id in flatten(self.lemma_pos_to_sense_ids[lemma].values()):
                counter[sense_id] += 1

        vocabulary = vocab(counter, min_freq=1, specials=[UNK_TOKEN])
        vocabulary.set_default_index(vocabulary[UNK_TOKEN])

        return vocabulary

    @staticmethod
    @lru_cache(maxsize=None)
    def get_wn_possible_sense_ids(lemma: str, pos: str) -> List[str]:
        lemmas: List[Lemma] = wordnet.lemmas(lemma, pos) if lemma else list()

        synsets = set()
        sense_ids = list()
        for lem in lemmas:
            # filter different sense ids which come from the same synset
            if lem.synset() not in synsets:
                synsets.add(lem.synset())
                sense_ids.append(lem.key())

        return sense_ids

    @lru_cache(maxsize=None)
    def __get_wn_gloss(self, sense_id: str) -> Tuple[str, str]:
        lemma: Lemma = wordnet.lemma_from_key(sense_id)
        return lemma.synset().definition(), lemma.name()


if __name__ == "__main__":

    training_corpus = read_wsd_corpus(f'{constants.TRAIN_SET_PATH}{constants.XML_DATA_SUFFIX}',
                                      f'{constants.TRAIN_SET_PATH}{constants.TXT_GOLD_KEYS_SUFFIX}')
    evaluation_corpus = read_wsd_corpus(f'{constants.TEST_SET_PATH}{constants.XML_DATA_SUFFIX}',
                                        f'{constants.TEST_SET_PATH}{constants.TXT_GOLD_KEYS_SUFFIX}')

    wic_corpus = list()
    for wic_sample in read_wic_corpus(constants.WIC_TEST_SET_PATH, constants.WIC_TEST_SET_WSD_KEYS_PATH):
        wic_corpus.append(wic_sample.sentence1)
        wic_corpus.append(wic_sample.sentence2)

    gloss_corpus = read_wsd_corpus(f"{constants.GLOSSES_PATH}{constants.XML_DATA_SUFFIX}",
                                   f"{constants.GLOSSES_PATH}{constants.TXT_GOLD_KEYS_SUFFIX}")

    pos_to_sense_ids = {"NOUN": list(), "VERB": list(), "ADJ": list(), "ADV": list()}
    lemma_pos_to_sense_ids = dict()

    samples = gloss_corpus + training_corpus + evaluation_corpus + wic_corpus
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        for tok in sample:
            if tok.is_tagged:
                lemma_pos_to_sense_ids[tok.lemma] = pos_to_sense_ids.copy()
                for pos in [Pos.NOUN, Pos.VERB, Pos.ADJ, Pos.ADV]:
                    wn_sense_ids = SenseInventory.get_wn_possible_sense_ids(tok.lemma, pos.to_wordnet())
                    lemma_pos_to_sense_ids[tok.lemma][str(pos)] = wn_sense_ids

    with open(constants.LEMMA_POS_DICT_PATH, "w") as file:
        json.dump(lemma_pos_to_sense_ids, file, indent=4)
