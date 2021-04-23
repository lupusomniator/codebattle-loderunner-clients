_ELEMENTS = dict(
    OUT_OF_WINDOW="!",
    NONE=" ",
    # walls
    BRICK="#",
    PIT_FILL_1="1",
    PIT_FILL_2="2",
    PIT_FILL_3="3",
    PIT_FILL_4="4",
    UNDESTROYABLE_WALL="☼",
    DRILL_PIT="*",
    ENEMY_LADDER="Q",
    ENEMY_LEFT="«",
    ENEMY_RIGHT="»",
    ENEMY_PIPE_LEFT="<",
    ENEMY_PIPE_RIGHT=">",
    ENEMY_PIT="X",
    YELLOW_GOLD="$",
    GREEN_GOLD="&",
    RED_GOLD="@",
    # This is your loderunner
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
    # this is other players
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
    # ladder and pipe - you can walk
    LADDER="H",
    PIPE="~",
    PORTAL="⊛",
    THE_SHADOW_PILL="S",
)

_ACTOR_ELEMENTS = dict(
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
    ENEMY_LADDER="Q",
    ENEMY_LEFT="«",
    ENEMY_RIGHT="»",
    ENEMY_PIPE_LEFT="<",
    ENEMY_PIPE_RIGHT=">",
    ENEMY_PIT="X",
)

_STATIC_ELEMENTS = {
    "BRICKS",
    "UNDESTROYABLE_WALL",
    "DRILL_PIT",
    "LADDER",
    "PIPE",
    "PORTAL"
}

ElementsCount = len(_ELEMENTS)

_INDEX_TO_ELEMENT = {}
_ELEMENT_TO_INDEX = {}

for i, el in enumerate(sorted(_ELEMENTS.values())):
    _ELEMENT_TO_INDEX[el] = i
    _INDEX_TO_ELEMENT[i] = el
