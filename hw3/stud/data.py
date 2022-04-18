import functools
from dataclasses import dataclass, asdict
from enum import Enum
from typing import *

from pytorch_lightning.utilities import AttributeDict


@dataclass
class HParams:
    num_classes: int
    input_size: int
    hidden_size: int = 100
    learning_rate: float = 1e-3

    def as_dict(self) -> AttributeDict:
        return AttributeDict(asdict(self))


class Pos(Enum):
    NOUN = "n"
    VERB = "v"
    ADJ = "a"
    ADV = "r"

    @classmethod
    def parse(cls, value) -> Optional["Pos"]:
        return cls[value] if value in cls._member_map_ else None

    def __str__(self) -> str:
        return self.name

    def to_wordnet(self) -> str:
        return self.value


@dataclass
class Token:
    text: str
    index: int
    id: Optional[str] = None
    sense_id: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[Pos] = None

    @property
    def wn_pos(self) -> Optional[str]:
        return self.pos.to_wordnet() if isinstance(self.pos, Pos) else None

    @property
    def is_tagged(self) -> bool:
        return self.sense_id is not None and self.id is not None


@dataclass
class WiCSample:
    sentence1: List[Token]
    sentence2: List[Token]
    label: bool
    __sense1: Token = None
    __sense2: Token = None

    @property
    def sense1(self) -> Token:
        if self.__sense1 is None:
            self.__sense1 = functools.reduce(lambda t1, t2: t1 if t1.id else t2, self.sentence1)
        return self.__sense1

    @property
    def sense2(self) -> Token:
        if self.__sense2 is None:
            self.__sense2 = functools.reduce(lambda t1, t2: t1 if t1.id else t2, self.sentence2)
        return self.__sense2
