from loderunnerclient.internals.element import Element
from typing import Iterable, Union, Optional, Tuple


# Actors


class AbstractActor:
    def __init__(self, element: Element):
        self.element = element


class Hero(AbstractActor):
    states = dict(
        HERO_DIE="Ѡ",
        HERO_DRILL_LEFT="Я",
        HERO_DRILL_RIGHT="R",
        HERO_LADDER="Y",
        HERO_LEFT="◄",
        HERO_RIGHT="►",
        HERO_FALL_LEFT="]",
        HERO_FALL_RIGHT="[",
        HERO_PIPE_LEFT="{",
        HERO_PIPE_RIGHT="}",
        HERO_SHADOW_DIE="x",
        HERO_SHADOW_DRILL_LEFT="⊰",
        HERO_SHADOW_DRILL_RIGHT="⊱",
        HERO_SHADOW_LADDER="⍬",
        HERO_SHADOW_LEFT="⊲",
        HERO_SHADOW_RIGHT="⊳",
        HERO_SHADOW_FALL_LEFT="⊅",
        HERO_SHADOW_FALL_RIGHT="⊄",
        HERO_SHADOW_PIPE_LEFT="⋜",
        HERO_SHADOW_PIPE_RIGHT="⋝",
    )

    def __init__(self,
                 is_left: bool,
                 is_on_pipe: bool,
                 is_on_ladder: bool,
                 is_drilling: bool,
                 is_falling: bool,
                 is_shadowed: bool,
                 is_dead: bool,
                 *args):
        super().__init__(*args)
        self.is_left = is_left
        self.is_on_pipe = is_on_pipe
        self.is_on_ladder = is_on_ladder
        self.is_drilling = is_drilling
        self.is_falling = is_falling
        self.is_shadowed = is_shadowed
        self.is_dead = is_dead
        self.name = "HERO"


class OtherHero(AbstractActor):
    states = dict(
        OTHER_HERO_DIE="Z",
        OTHER_HERO_DRILL_LEFT="⌋",
        OTHER_HERO_DRILL_RIGHT="⌊",
        OTHER_HERO_LEFT=")",
        OTHER_HERO_RIGHT="(",
        OTHER_HERO_FALL_LEFT="⊐",
        OTHER_HERO_FALL_RIGHT="⊐",
        OTHER_HERO_LADDER="U",
        OTHER_HERO_PIPE_LEFT="Э",
        OTHER_HERO_PIPE_RIGHT="Є",

        OTHER_HERO_SHADOW_DIE="⋈",
        OTHER_HERO_SHADOW_DRILL_LEFT="⋰",
        OTHER_HERO_SHADOW_DRILL_RIGHT="⋱",
        OTHER_HERO_SHADOW_LEFT="⋊",
        OTHER_HERO_SHADOW_RIGHT="⋉",
        OTHER_HERO_SHADOW_LADDER="⋕",
        OTHER_HERO_SHADOW_FALL_LEFT="⋣",
        OTHER_HERO_SHADOW_FALL_RIGHT='⋢',
        OTHER_HERO_SHADOW_PIPE_LEFT="⊣",
        OTHER_HERO_SHADOW_PIPE_RIGHT="⊢",
    )

    def __init__(self,
                 is_left: bool,
                 is_on_pipe: bool,
                 is_on_ladder: bool,
                 is_shadowed: bool,
                 *args):
        super().__init__(*args)
        self.is_left = is_left
        self.is_on_pipe = is_on_pipe
        self.is_on_ladder = is_on_ladder
        self.is_shadowed = is_shadowed
        self.name = "OTHER_HERO"


class Enemy(AbstractActor):
    states = dict(
        ENEMY_LADDER="Q",
        ENEMY_LEFT="«",
        ENEMY_RIGHT="»",
        ENEMY_PIPE_LEFT="<",
        ENEMY_PIPE_RIGHT=">",
        ENEMY_PIT="X",
    )

    def __init__(self,
                 is_left: bool,
                 is_on_pipe: bool,
                 is_on_ladder: bool,
                 is_in_pit: bool,
                 *args
                 ):
        super().__init__(*args)
        self.is_left = is_left
        self.is_on_pipe = is_on_pipe
        self.is_on_ladder = is_on_ladder
        self.is_in_pit = is_in_pit
        self.name = "ENEMY"


def is_actor(value: Union[str, Element]):
    if isinstance(value, str):
        elem = Element(value)
    else:
        elem = value
    name = elem.get_name()
    return name in Hero.states or name in OtherHero.states or name in Enemy.states


def is_dangerous_actor(actor: AbstractActor):
    if isinstance(actor, Enemy):
        return True
    if isinstance(actor, OtherHero) and actor.is_shadowed:
        return True
    return False

def is_not_dangerous_actor


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
