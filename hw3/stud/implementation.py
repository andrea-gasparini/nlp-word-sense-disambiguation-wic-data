from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from model import Model

from stud import utils
from stud.data_readers import parse_wic_samples
from stud.datasets import GlossBERTDataset
from stud.pl_modules import GlossBERT
from stud.sense_inventories import SenseInventory


def build_model(device: str) -> Model:
    return StudentModel(device)


class StudentModel(Model):
    
    def __init__(self, device: str):
        # load NLTK stuff
        utils.nltk_downloads()

        # load sense inventory and vocabulary
        self.sense_inventory = SenseInventory("model/glosses/glosses_main", "model/lemma_pos_dictionary.json")
        self.vocab: Vocab = torch.load("model/vocabularies/senses_vocabulary_full.pt")
        self.vocab_itos = self.vocab.get_itos()

        self.model = GlossBERT.load_from_checkpoint("model/GlossBERT.ckpt", map_location=device,
                                                    bert_model_name_or_path="model/bert-base-cased")
        self.hparams = self.model.hparams
        self.model.freeze()

    def predict(self, sentence_pairs: List[Dict]) -> Tuple[List[str], List[Tuple[str, str]]]:

        wic_samples = parse_wic_samples(sentence_pairs)
        wic_corpus = list()
        for wic_sample in wic_samples:
            wic_corpus.append(wic_sample.sentence1)
            wic_corpus.append(wic_sample.sentence2)

        dataset = GlossBERTDataset.from_tokens(wic_corpus, self.sense_inventory, ignore_pos=False)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, collate_fn=GlossBERTDataset.collate_fn)

        predictions = list()
        tokens, probabilities, sense_indices = list(), list(), list()

        for batch in dataloader:
            out = self.model(batch)
            for i in range(len(batch["tokens"])):
                token = batch["tokens"][i]
                prob = out["probabilities"][i]
                sense_index = batch["sense_indices"][i]
                if len(tokens) > 0 and token == tokens[-1][-1]:
                    tokens[-1].append(token)
                    probabilities[-1] = torch.cat((probabilities[-1], torch.tensor([prob])))
                    sense_indices[-1] = torch.cat((sense_indices[-1], torch.tensor([sense_index])))
                else:
                    tokens.append([token])
                    probabilities.append(torch.tensor([prob]))
                    sense_indices.append(torch.tensor([sense_index]))

        for i in range(len(probabilities)):
            sample_prediction_index = torch.argmax(probabilities[i])
            prediction_index = sense_indices[i][sample_prediction_index]
            predictions.append(self.vocab_itos[prediction_index])

        # build WSD pairs predictions as the grader expects
        predictions_wsd = [(predictions[i], predictions[i + 1]) for i in range(0, len(predictions), 2)]
        predictions_wic = [str(prediction_wsd[0] == prediction_wsd[1]) for prediction_wsd in predictions_wsd]

        return predictions_wic, predictions_wsd
