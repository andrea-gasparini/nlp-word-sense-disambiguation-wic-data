import os.path
from collections import defaultdict
from typing import *
from xml.etree import ElementTree

from jsonlines import jsonlines
from nltk import TreebankWordTokenizer

from stud.data import Token, Pos, WiCSample


def read_wsd_gold_keys(txt_path: str) -> Dict[str, str]:
    """
    Reads the gold keys of a WSD corpus from a txt file
    and parses it into a dictionary that goes from tokens ids to wordnet sense ids.

    Args:
        txt_path: gold keys file

    Returns: tokens ids to wordnet sense ids dictionary
    """
    if not os.path.isfile(txt_path):
        raise Exception(f"{txt_path} is not a valid txt gold keys file")

    with open(txt_path) as f:
        gold_keys = [line.strip().split(' ') for line in f]

    sense_ids_dict = dict()
    for gold_key in gold_keys:
        token_id = gold_key[0]
        sense_id = gold_key[1]  # ignore eventual secondary senses ([2:])
        sense_ids_dict[token_id] = sense_id

    return sense_ids_dict


def read_wsd_corpus(xml_data_path: str, txt_gold_keys_path: str) -> List[List[Token]]:
    """
    Parses a WSD corpus from reading an xml data file and a txt gold keys file.

    Args:
        xml_data_path: WSD data file
        txt_gold_keys_path: WSD gold keys file

    Returns: a list of parsed sentences (list of `Token`)
    """
    if not os.path.isfile(xml_data_path):
        raise Exception(f"{xml_data_path} is not a valid xml data file")

    sense_ids_dict = read_wsd_gold_keys(txt_gold_keys_path)

    sentences = list()

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


def read_wic_corpus(jsonl_data_path: str, wsd_txt_gold_keys_path: Optional[str] = None) -> List[WiCSample]:
    """
    Parses a WiC corpus from reading a jsonl data file and, optionally, a txt file for WSD gold keys.

    Args:
        jsonl_data_path: WiC data file
        wsd_txt_gold_keys_path: optional WSD gold keys file

    Returns: a list of parsed `WiCSample`
    """
    if not os.path.isfile(jsonl_data_path):
        raise Exception(f"{jsonl_data_path} is not a valid jsonl data file")

    sense_ids_dict = read_wsd_gold_keys(wsd_txt_gold_keys_path) if wsd_txt_gold_keys_path is not None else None

    samples = list()

    tokenizer = TreebankWordTokenizer()

    with jsonlines.open(jsonl_data_path) as f:
        for raw_sample in f:
            raw_sample: Dict = defaultdict(lambda: None, raw_sample)
            encoded_sentences = (list(), list())

            # iterate over sentence numbers (i.e. sentence1 and sentence2)
            for sentence_n in [1, 2]:

                sentence = raw_sample[f"sentence{sentence_n}"]
                tokenized_sentence = tokenizer.tokenize(sentence)
                target_start_index = int(raw_sample[f"start{sentence_n}"])
                target_previous_tokens = len(tokenizer.tokenize(sentence[:target_start_index]))

                for token_index in range(len(tokenized_sentence)):
                    token = Token(text=tokenized_sentence[token_index].lower(), index=token_index)

                    if token_index == target_previous_tokens:
                        token.lemma = raw_sample["lemma"]
                        token.pos = Pos.parse(raw_sample["pos"])
                        token.id = f"{raw_sample['id']}.s{sentence_n}"

                        if sense_ids_dict is not None:
                            token.sense_id = sense_ids_dict[token.id]

                    encoded_sentences[sentence_n - 1].append(token)

            sample = WiCSample(sentence1=encoded_sentences[0],
                               sentence2=encoded_sentences[1],
                               label=bool(raw_sample["label"]))
            samples.append(sample)

    return samples
