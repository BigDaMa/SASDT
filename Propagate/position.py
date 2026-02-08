from abc import abstractmethod
from typing import Literal, Optional



class Position:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_index(self, v: str) -> int:
        pass

class AbsPosPosition(Position):

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def get_index(self, v: str) -> Optional[int]:
        return self.idx
    
    def __str__(self) -> str:
        return f"AbsPosPosition({self.idx})"

    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(("AbsPosPosition", self.idx))

    def __eq__(self, other):
        if not isinstance(other, AbsPosPosition):
            return False
        return self.idx == other.idx

class RegExPosPosition(Position):
    def __init__(self, regex: str, occurance: int, direction: Literal["L", "R"]):
        super().__init__()
        self.regex = regex
        self.occurance = occurance
        self.direction = direction

    def get_regex(self) -> str:
        return self.regex

    def get_index(self, v: str) -> int:
        from Propagate.positionop import RegExPos
        return RegExPos(v, self.regex, self.occurance, self.direction)
    
    def get_as_tuple(self) -> tuple:
        return (self.regex, self.lidx, self.ridx)
    
    def __str__(self) -> str:
        return f"RegExPosPosition({self.regex}, {self.occurance}, {self.direction})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self.regex, self.occurance, self.direction))
    
    def __eq__(self, other):
        if not isinstance(other, RegExPosPosition):
            return False
        return (self.regex, self.occurance, self.direction) == (other.regex, other.occurance, other.direction)