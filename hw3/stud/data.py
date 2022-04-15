from dataclasses import dataclass
from enum import Enum
from typing import *


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
