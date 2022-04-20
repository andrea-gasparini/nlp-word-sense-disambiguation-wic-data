UNK_TOKEN: str = "[UNK]"
PAD_TOKEN: str = "[PAD]"
PAD_INDEX: int = 0

TRANSFORMER_EMBEDDER_MODEL: str = "bert-base-cased"
TRANSFORMER_EMBEDDER_SEP_TOKEN: str = "[SEP]"
TRANSFORMER_EMBEDDER_PATH: str = f"../../model/{TRANSFORMER_EMBEDDER_MODEL}"

GLOSSES_PATH: str = "../../data/glosses/glosses_main"

LEMMA_POS_DICT_PATH = "../../data/lemma_pos_dictionary.json"

TRAIN_SET_PATH: str = "../../data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor"
VALID_SET_PATH: str = "../../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007"
TEST_SET_PATH: str = "../../data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL"

WIC_TEST_SET_PATH: str = "../../data/dev.jsonl"
WIC_TEST_SET_WSD_KEYS_PATH: str = "../../data/dev_wsd.txt"
WIC_TRAIN_SET_PATH: str = "../../data/train.jsonl"

PREPROCESSED_TRAIN_PATH: str = "../../data/preprocessed/WSDisambiguator/SemCor.pt"
PREPROCESSED_VALID_PATH: str = "../../data/preprocessed/WSDisambiguator/semeval2007.pt"
PREPROCESSED_TEST_PATH: str = "../../data/preprocessed/WSDisambiguator/wic_dev.pt"
PREPROCESSED_GLOSSBERT_TRAIN_PATH: str = "../../data/preprocessed/GlossBERT/SemCor10+wic_train.json"
PREPROCESSED_GLOSSBERT_VALID_PATH: str = "../../data/preprocessed/GlossBERT/semeval2007.json"
PREPROCESSED_GLOSSBERT_TEST_PATH: str = "../../data/preprocessed/GlossBERT/wic_dev.json"

XML_DATA_SUFFIX: str = ".data.xml"
TXT_GOLD_KEYS_SUFFIX: str = ".gold.key.txt"
