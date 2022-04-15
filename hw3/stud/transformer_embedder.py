from typing import *

import torch
from torch import Tensor
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, PreTrainedTokenizer

from stud import utils


class TransformerEmbedder(torch.nn.Module):

    def __init__(self, pretrained_model_name_or_path: str, device: str = "cpu") -> None:
        super().__init__()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model: PreTrainedModel = AutoModel.from_pretrained(pretrained_model_name_or_path,
                                                                output_hidden_states=True,
                                                                output_attentions=True)
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.embedding_dimension = self.model.config.hidden_size

    def forward(self, tokens_batch: List[List[str]]) -> Tensor:
        encoding = self.tokenizer(tokens_batch,
                                  padding=True,
                                  truncation=False,
                                  return_tensors="pt",
                                  is_split_into_words=True)

        input_ids = encoding["input_ids"].to(self.device)

        with torch.no_grad():
            # shape: (batch_size, num_sub-words, embedding_size)
            encoder_outputs = self.model(input_ids)

        # layer pooling: last
        pooled_output = encoder_outputs.last_hidden_state

        # sub-words pooling: mean
        word_ids = [sample.word_ids for sample in encoding.encodings]
        aggregated_wp_output = [aggregate_subword_vectors(sample_word_ids, sample_vectors) for
                                sample_word_ids, sample_vectors in zip(word_ids, pooled_output)]
        pooled_output = utils.pad_sequence([merge_subwords_vectors(pairs) for pairs in aggregated_wp_output])

        return remove_transformer_tokens(pooled_output, tokens_batch)


def remove_transformer_tokens(encodings, tokens_batch: List[List[str]]) -> Tensor:
    encoding_mask = list()

    for tokens in tokens_batch:
        sample_mask = len(tokens) * [True]
        # add False for both [CLS] and [SEP]
        sample_mask = [False] + sample_mask + [False]
        # add False as [PAD] to match the padded batch len
        padded_sample_mask = sample_mask + [False] * (encodings.shape[1] - len(sample_mask))
        encoding_mask.append(torch.tensor(padded_sample_mask))

    encoding_mask = torch.stack(encoding_mask)

    flattened_filtered_encodings = encodings[encoding_mask]
    encodings = flattened_filtered_encodings.split([len(tokens) for tokens in tokens_batch])
    return utils.pad_sequence(encodings)


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
