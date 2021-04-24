from enum import Enum

class AutoNumber(Enum):
    def __new__(cls, *args):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj.num = value
        return obj

class LoderunnerAction(AutoNumber):
    GO_LEFT = "left"
    GO_RIGHT = "right"
    GO_UP = "up"
    GO_DOWN = "down"
    DRILL_RIGHT = "act,right"
    DRILL_LEFT = "act,left"
    DO_NOTHING = "stop"
    SUICIDE = "act(0)"
    FILL_PIT = "fill_pit"

def get_action_by_num(num):
    return list(LoderunnerAction)[num]

