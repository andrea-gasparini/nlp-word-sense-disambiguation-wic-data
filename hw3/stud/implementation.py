import numpy as np
from typing import List, Tuple, Dict

from model import Model
from nltk.corpus import wordnet as wn
import nltk
nltk.download("wordnet")

mapping = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> Tuple[List[str], List[str]]:
        preds_wsd = [(np.random.choice(wn.synsets(pair["lemma"], mapping[pair["pos"]])).lemmas()[0].key(), np.random.choice(wn.synsets(pair["lemma"], mapping[pair["pos"]])).lemmas()[0].key()) for pair in sentence_pairs]
        preds_wic = []
        for pred in preds_wsd:
            if pred[0] == pred[1]:
                preds_wic.append('True')
            else:
                preds_wic.append('False')
        return preds_wic, preds_wsd


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!
        pass
