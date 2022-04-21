from typing import *

import torch
import pytorch_lightning as pl
from torch import Tensor
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, PreTrainedTokenizer

from stud import utils


class TransformerEmbedder(pl.LightningModule):
    """
    A wrapper class for Transformer based encoder models (e.g. BERT).
    It takes a pretrained Hugging Face model name (or a path to its local dump) and can be either fine-tuned or not.

    Given a batch of tokenized sentences, it returns the hidden states of the last layer,
    averaging eventual WordPiece sub-words representations into one.
    """

    def __init__(self, pretrained_model_name_or_path: str, fine_tune: bool = False) -> None:
        super().__init__()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model: PreTrainedModel = AutoModel.from_pretrained(pretrained_model_name_or_path,
                                                                output_hidden_states=True,
                                                                output_attentions=True)
        if not fine_tune:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.embedding_dimension = self.model.config.hidden_size

    def forward(self, tokens_batch: List[List[str]], device: Optional[str] = None) -> Tensor:
        encoding = self.tokenizer(tokens_batch,
                                  padding=True,
                                  truncation=False,
                                  return_tensors="pt",
                                  is_split_into_words=True)

        self.model.to(self.device if device is None else device)
        encoder_in = {k: v.to(self.device if device is None else device) for k, v in encoding.items()}

        # shape: (batch_size, num_sub-words, embedding_size)
        encoder_outputs = self.model(**encoder_in)

        # layer pooling: last
        pooled_output = encoder_outputs.last_hidden_state

        # sub-words pooling: mean
        word_ids = [sample.word_ids for sample in encoding.encodings]
        aggregated_wp_output = [aggregate_subword_vectors(sample_word_ids, sample_vectors) for
                                sample_word_ids, sample_vectors in zip(word_ids, pooled_output)]

        return utils.pad_sequence([merge_subwords_vectors(pairs) for pairs in aggregated_wp_output])


def aggregate_subword_vectors(word_ids: List[int], vectors: Tensor) -> List[List[Tuple[int, Tensor]]]:
    aggregated_tokens = list()
    token = [(word_ids[0], vectors[0])]

    for w_id, vector in zip(word_ids[1:], vectors[1:]):
        vector = vector
        if w_id is not None and w_id == token[-1][0]:
            token.append((w_id, vector))
        else:
            aggregated_tokens.append(token)
            token = [(w_id, vector)]

    if len(token) > 0:
        aggregated_tokens.append(token)

    return aggregated_tokens


def merge_subwords_vectors(subword_vector_pairs: List[List[Tuple[int, Tensor]]]) -> Tensor:
    vectors = list()
    for pairs in subword_vector_pairs:
        _, pair_vectors = zip(*pairs)
        vector = torch.stack(pair_vectors).mean(dim=0)
        vectors.append(vector)
    return torch.stack(vectors)
