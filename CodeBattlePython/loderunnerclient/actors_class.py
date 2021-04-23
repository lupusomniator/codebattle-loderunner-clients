from loderunnerclient.internals.constants import *

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
        OTHER_HERO_LEFT=")",
        OTHER_HERO_RIGHT="(",
        OTHER_HERO_LADDER="U",
        OTHER_HERO_PIPE_LEFT="Э",
        OTHER_HERO_PIPE_RIGHT="Є",
        OTHER_HERO_SHADOW_DIE="⋈",
        OTHER_HERO_SHADOW_LEFT="⋊",
        OTHER_HERO_SHADOW_RIGHT="⋉",
        OTHER_HERO_SHADOW_LADDER="⋕",
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
