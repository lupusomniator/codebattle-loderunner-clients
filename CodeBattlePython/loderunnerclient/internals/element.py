from loderunnerclient.internals.constants import (_ELEMENTS, _HERO_ELEMENTS,
                                                  _ACTOR_ELEMENTS, _ELEMENT_TO_INDEX,
                                                  _INDEX_TO_ELEMENT, _STATIC_ELEMENTS,
                                                  _ENEMY_ELEMENTS, _HEROES_TO_FALL_MAP,
                                                  _HEROES_TO_PIPE_MAP)
from typing import Iterable, Union, Optional, Tuple


def index_to_char(index):
    return _INDEX_TO_ELEMENT[index]


def char_to_index(char):
    if char in _ELEMENT_TO_INDEX:
        return _ELEMENT_TO_INDEX[char]
    else:
        return _ELEMENT_TO_INDEX[" "]


def value_of(char):
    """ Test whether the char is valid Element and return it's name."""
    for value, c in _ELEMENTS.items():
        if char == c:
            return value
    else:
        raise AttributeError("No such Element: {}".format(char))


def char_of(name):
    for value, c in _ELEMENTS.items():
        if name == value:
            return c
    else:
        raise AttributeError("No such Element: {}".format(name))


class Element:
    """ Class describes the Element objects for Bomberman game."""

    def __init__(self, n_or_c):
        """ Construct an Element object from given name or char."""
        for n, c in _ELEMENTS.items():
            if n_or_c == n or n_or_c == c:
                self._name = n
                self._char = c
                break
        else:
            raise AttributeError("No such Element: {}".format(n_or_c))

    def get_char(self):
        """ Return the Element's character."""
        return self._char

    def get_name(self):
        """ Return the Element's character."""
        return self._name

    def __eq__(self, otherElement):
        return self._name == otherElement._name and self._char == otherElement._char

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()


def to_element(value: Union[str, Element]):
    if isinstance(value, str):
        return Element(value)
    else:
        return value


def is_actor(value: Union[str, Element]):
    return to_element(value).get_name() in _ACTOR_ELEMENTS


def is_hero(value: Union[str, Element]):
    return to_element(value).get_name() in _HERO_ELEMENTS


def is_holding_actor(value: Union[str, Element]):
    el = to_element(value)
    name = el.get_name()
    return is_actor(el) and ("PIPE" in name or "LADDER" in name)

def is_gold(value: Union[str, Element]):
    el = to_element(value)
    return "GOLD" in el.get_name()

def is_enemy(value: Union[str, Element]):
    return to_element(value).get_name() in _ENEMY_ELEMENTS

def is_pit_fill(value: Union[str, Element]):
    return "PIT_FILL" in to_element(value).get_name()

def to_falling(value: Union[str, Element]):
    el = to_element(value)
    if el.get_name() in _HEROES_TO_FALL_MAP:
        return Element(_HEROES_TO_FALL_MAP[el.get_name()])

    return el

def to_ladder(value: Union[str, Element]):
    el = to_element(value)
    if 'OTHER_HERO' in el.get_name():
        return Element('OTHER_HERO_LADDER')

    if 'HERO' in el.get_name():
        return Element('HERO_LADDER')

    if 'ENEMY' in el.get_name():
        return Element('ENEMY_LADDER')

def to_pipe(value: Union[str, Element]):
    el = to_element(value)
    if el.get_name() in _HEROES_TO_PIPE_MAP:
        return Element(_HEROES_TO_PIPE_MAP[el.get_name()])

    return el


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be ran from CLI")
