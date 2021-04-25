from loderunnerclient.internals.element import Element
from typing import Iterable, Union, Optional, Tuple

_ENTITY_ELEMENTS = dict(
    GREEN_GOLD="&",
    YELLOW_GOLD="$",
    RED_GOLD="@",
    SHADOW_PILL="S",
    PORTAL="⊛"
)

# Entities

class AbstractEntity:
    def __init__(self, element: Element):
        self.element = element
        self.reward = -0.1

    def get_reward(self):
        return self.reward


class GoldGreen(AbstractEntity):
    states = dict(
        GREEN_GOLD="&",
    )

    def __init__(self, *args):
        super().__init__(*args)
        self.reward = 1
        self.name = "GREEN_GOLD"


class GoldYellow(AbstractEntity):
    states = dict(
        YELLOW_GOLD="$",
    )

    def __init__(self, *args):
        super().__init__(*args)
        self.reward = 2
        self.name = "YELLOW_GOLD"


class GoldRed(AbstractEntity):
    states = dict(
        RED_GOLD="@",
    )

    def __init__(self, *args):
        super().__init__(*args)
        self.reward = 3
        self.name = "RED_GOLD"


class Pill(AbstractEntity):
    states = dict(
        SHADOW_PILL="S"
    )

    def __init__(self, *args):
        super().__init__(*args)
        self.name = "SHADOW_PILL"


class Portal(AbstractEntity):
    states = dict(
        PORTAL="⊛"
    )

    def __init__(self, *args):
        super().__init__(*args)
        self.name = "PORTAL"
        self.reward = 0.5


def is_entity(value: Union[str, Element]):
    if isinstance(value, str):
        elem = Element(value)
    else:
        elem = value
    return elem.get_name() in _ENTITY_ELEMENTS


def create_entity(value: Union[str, Element]):
    if isinstance(value, str):
        elem = Element(value)
    else:
        elem = value
    name = elem.get_name()
    if name in GoldGreen.states:
        return GoldGreen(elem)
    if name in GoldYellow.states:
        return GoldYellow(elem)
    if name in GoldRed.states:
        return GoldRed(elem)
    if name in Pill.states:
        return Pill(elem)
