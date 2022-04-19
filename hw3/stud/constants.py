UNK_TOKEN: str = "[UNK]"
PAD_TOKEN: str = "[PAD]"
PAD_INDEX: int = 0

TRANSFORMER_EMBEDDER_MODEL: str = "bert-base-cased"
TRANSFORMER_EMBEDDER_PATH: str = f"../../model/{TRANSFORMER_EMBEDDER_MODEL}"

GLOSSES_PATH: str = "../../data/glosses/glosses_main"

LEMMA_POS_DICT_PATH = "../../data/preprocessed/lemma_pos_dictionary.json"

TRAIN_SET_PATH: str = "../../data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor"
VALID_SET_PATH: str = "../../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007"
TEST_SET_PATH: str = "../../data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL"

WIC_TEST_SET_PATH: str = "../../data/dev.jsonl"
WIC_TEST_SET_WSD_KEYS_PATH: str = "../../data/dev_wsd.txt"

PREPROCESSED_TRAIN_PATH: str = "../../data/preprocessed/SemCor_preprocessed.pt"
PREPROCESSED_VALID_PATH: str = "../../data/preprocessed/semeval2007_preprocessed.pt"
PREPROCESSED_TEST_PATH: str = "../../data/preprocessed/WSD_evaluation_all_preprocessed.pt"

XML_DATA_SUFFIX: str = ".data.xml"
TXT_GOLD_KEYS_SUFFIX: str = ".gold.key.txt"
