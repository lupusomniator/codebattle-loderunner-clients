import numpy as np
import loderunnerclient.internals.element as element
from typing import Iterable, Union, Optional, Tuple

all_elements = element._ELEMENTS

_SPACE_ELEMENTS = dict(
    NONE=" ",
    # walls
    BRICK="#",
    DRILL_PIT="*",
    PIT_FILL_1="1",
    PIT_FILL_2="2",
    PIT_FILL_3="3",
    PIT_FILL_4="4",
    UNDESTROYABLE_WALL="☼",
    LADDER="H",
    PIPE="~"
)


class Direction:
    left = np.array([-1, 0])
    right = np.array([1, 0])
    down = np.array([0, 1])
    up = np.array([0, -1])


class AbstractNode:
    states: dict = {}

    def __init__(self, elem: element.Element, **kwargs):
        self.element = elem
        self.available_moves: np.ndarray = np.array([])


class EmptySpace(AbstractNode):
    states = dict(
        NONE=" "
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_moves = np.array([
            Direction.down,
            Direction.left,
            Direction.right
        ])
        self.name = "NONE"


class UndestroyableWall(AbstractNode):
    states = dict(
        UNDESTROYABLE_WALL="☼"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "UNDESTROYABLE_WALL"


class Brick(AbstractNode):
    states = dict(
        BRICK="#",
        DRILL_PIT="*",
        PIT_FILL_1="1",
        PIT_FILL_2="2",
        PIT_FILL_3="3",
        PIT_FILL_4="4"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BRICK"


class Ladder(AbstractNode):
    states = dict(
        LADDER="H"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_moves = np.array([
            Direction.down,
            Direction.left,
            Direction.right,
            Direction.up
        ])
        self.name = "LADDER"


class Pipe(AbstractNode):
    states = dict(
        PIPE="~"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_moves = np.array([
            Direction.down,
            Direction.left,
            Direction.right
        ])
        self.name = "PIPE"


def is_space(value: Union[str, element.Element]):
    if isinstance(value, str):
        elem = element.Element(value)
    else:
        elem = value
    return elem.get_name() in _SPACE_ELEMENTS


def is_available_space(value: Union[str, element.Element, AbstractNode]):
    if isinstance(value, AbstractNode):
        if isinstance(value, EmptySpace) or isinstance(value, Ladder) or isinstance(value, Pipe):
            return True
        return False
    if isinstance(value, str):
        elem = element.Element(value)
    else:
        elem = value
    name = elem.get_name()
    return name in Pipe.states or name in Ladder.states or name in EmptySpace.states


def create_space_element(value: Union[str, element.Element]):
    if isinstance(value, str):
        elem = element.Element(value)
    else:
        elem = value
    name = elem.get_name()
    if name in EmptySpace.states:
        return EmptySpace(elem)
    elif name in UndestroyableWall.states:
        return UndestroyableWall(elem)
    elif name in Brick.states:
        return Brick(elem)
    elif name in Ladder.states:
        return Ladder(elem)
    elif name in Pipe.states:
        return Pipe(elem)
