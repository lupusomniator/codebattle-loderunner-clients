_ELEMENTS = dict(
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
    ENEMY_LADDER="Q",
    ENEMY_LEFT="«",
    ENEMY_RIGHT="»",
    ENEMY_PIPE_LEFT="<",
    ENEMY_PIPE_RIGHT=">",
    ENEMY_PIT="X",
    GREEN_GOLD="&",
    YELLOW_GOLD="$",
    RED_GOLD="@",
    SHADOW_PILL="S",
    PORTAL="⊛",
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
    PIPE="~",
)

_HERO_ELEMENTS = dict(
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
)

_ENEMY_ELEMENTS = dict(
    ENEMY_LADDER="Q",
    ENEMY_LEFT="«",
    ENEMY_RIGHT="»",
    ENEMY_PIPE_LEFT="<",
    ENEMY_PIPE_RIGHT=">",
    ENEMY_PIT="X",
)
_ACTOR_ELEMENTS = _HERO_ELEMENTS.copy()
_ACTOR_ELEMENTS.update(_ENEMY_ELEMENTS)

_STATIC_ELEMENTS = {
    "BRICK",
    "UNDESTROYABLE_WALL",
    "DRILL_PIT",
    "LADDER",
    "PIPE",
    "PORTAL"
}

_ELEMENTS_CAN_FLIED = {
    'NONE',
    'YELLOW_GOLD',
    'GREEN_GOLD',
    'RED_GOLD'
}

_HEROES_ON_STATIC_ELEMENTS_MAP = {
    'HERO_LADDER': 'LADDER',
    'HERO_SHADOW_LADDER': 'LADDER',
    'ENEMY_LADDER': 'LADDER',
    'ENEMY_SHADOW_LADDER': 'LADDER',
    'OTHER_HERO_LADDER': 'LADDER',
    'OTHER_HERO_SHADOW_LADDER': 'LADDER',

    'HERO_PIPE_LEFT': 'PIPE',
    'HERO_SHADOW_PIPE_LEFT': 'PIPE',
    'ENEMY_PIPE_LEFT': 'PIPE',
    'ENEMY_SHADOW_PIPE_LEFT': 'PIPE',
    'OTHER_HERO_PIPE_LEFT': 'PIPE',
    'OTHER_HERO_SHADOW_PIPE_LEFT': 'PIPE',

    'HERO_PIPE_RIGHT': 'PIPE',
    'HERO_SHADOW_PIPE_RIGHT': 'PIPE',
    'ENEMY_PIPE_RIGHT': 'PIPE',
    'ENEMY_SHADOW_PIPE_RIGHT': 'PIPE',
    'OTHER_HERO_PIPE_RIGHT': 'PIPE',
    'OTHER_HERO_SHADOW_PIPE_RIGHT': 'PIPE'
}

_HEROES_TO_FALL_MAP = {
    'HERO_LEFT': 'HERO_FALL_LEFT',
    'HERO_RIGHT': 'HERO_FALL_RIGHT',
    'ENEMY_LEFT': 'ENEMY_FALL_LEFT',
    'ENEMY_RIGHT': 'ENEMY_FALL_RIGHT',
    'OTHER_HERO_LEFT': 'OTHER_HERO_FALL_LEFT',
    'OTHER_HERO_RIGHT': 'OTHER_HERO_FALL_RIGHT',

    'HERO_LADDER': 'HERO_FALL_RIGHT',
    'HERO_SHADOW_LADDER': 'HERO_SHADOW_FALL_RIGHT',
    'ENEMY_LADDER': 'ENEMY_FALL_RIGHT',
    'ENEMY_SHADOW_LADDER': 'ENEMY_SHADOW_FALL_RIGHT',
    'OTHER_HERO_LADDER': 'OTHER_HERO_FALL_RIGHT',
    'OTHER_HERO_SHADOW_LADDER': 'OTHER_HERO_SHADOW_FALL_RIGHT',

    'HERO_PIPE_LEFT': 'HERO_FALL_LEFT',
    'HERO_SHADOW_PIPE_LEFT': 'HERO_SHADOW_FALL_LEFT',
    'ENEMY_PIPE_LEFT': 'ENEMY_FALL_LEFT',
    'ENEMY_SHADOW_PIPE_LEFT': 'ENEMY_SHADOW_FALL_LEFT',
    'OTHER_HERO_PIPE_LEFT': 'OTHER_HERO_FALL_LEFT',
    'OTHER_HERO_SHADOW_PIPE_LEFT': 'OTHER_HERO_SHADOW_FALL_RIGHT',

    'HERO_PIPE_RIGHT': 'HERO_FALL_RIGHT',
    'HERO_SHADOW_PIPE_RIGHT': 'HERO_SHADOW_FALL_RIGHT',
    'ENEMY_PIPE_RIGHT': 'ENEMY_FALL_RIGHT',
    'ENEMY_SHADOW_PIPE_RIGHT': 'ENEMY_SHADOW_FALL_RIGHT',
    'OTHER_HERO_PIPE_RIGHT': 'OTHER_HERO_FALL_RIGHT',
    'OTHER_HERO_SHADOW_PIPE_RIGHT': 'OTHER_HERO_SHADOW_FALL_RIGHT'
}

_HEROES_TO_PIPE_MAP = {
    'HERO_LEFT': 'HERO_PIPE_LEFT',
    'HERO_RIGHT': 'HERO_PIPE_RIGHT',
    'ENEMY_LEFT': 'ENEMY_PIPE_LEFT',
    'ENEMY_RIGHT': 'ENEMY_PIPE_RIGHT',
    'OTHER_HERO_LEFT': 'OTHER_HERO_PIPE_LEFT',
    'OTHER_HERO_RIGHT': 'OTHER_HERO_PIPE_RIGHT',
    'HERO_FALL_RIGHT': 'HERO_PIPE_RIGHT',
    'HERO_FALL_LEFT': 'HERO_PIPE_LEFT',
    'ENEMY_FALL_RIGHT': 'ENEMY_PIPE_RIGHT',
    'ENEMY_FALL_LEFT': 'ENEMY_PIPE_LEFT',
    'OTHER_HERO_FALL_RIGHT': 'OTHER_HERO_PIPE_RIGHT',
    'OTHER_HERO_FALL_LEFT': 'OTHER_HERO_PIPE_LEFT',
}

ElementsCount = len(_ELEMENTS)

_INDEX_TO_ELEMENT = {}
_ELEMENT_TO_INDEX = {}

for i, el in enumerate(sorted(_ELEMENTS.values())):
    _ELEMENT_TO_INDEX[el] = i
    _INDEX_TO_ELEMENT[i] = el
