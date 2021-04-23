from loderunnerclient.internals.constants import *
from loderunnerclient.internals.element import Element
from typing import Iterable, Union, Optional, Tuple


# Actors


def is_actor(value: Union[str, Element]):
    if isinstance(value, str):
        elem = Element(value)
    else:
        elem = value
    return elem.get_name() in _ACTOR_ELEMENTS


def create_actor(value: Union[str, Element]):
    if isinstance(value, str):
        elem = Element(value)
    else:
        elem = value
    name = elem.get_name()

    is_left = "LEFT" in name
    is_on_pipe = "PIPE" in name
    is_on_ladder = "LADDER" in name
    is_drilling = "DRILL" in name
    is_falling = "FALL" in name
    is_shadowed = "SHADOW" in name
    is_dead = "DIE" in name
    is_in_pit = "PIT" in name
    if name in Hero.states:
        return Hero(is_left, is_on_pipe, is_on_ladder, is_drilling, is_falling, is_shadowed,
                    is_dead, elem)
    if name in OtherHero.states:
        return OtherHero(is_left, is_on_pipe, is_on_ladder, is_shadowed, elem)
    if name in Enemy.states:
        return Enemy(is_left, is_on_pipe, is_on_ladder, is_in_pit, elem)
