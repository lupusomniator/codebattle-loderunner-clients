from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.element import Element
from loderunnerclient.internals.constants import _ELEMENTS_CAN_FLIED
from enum import Enum
import random

class MapChangeType(Enum):
    NONE = 0,
    CHANGE = 1,
    MOVE_OR_INTERACT = 2

class MapChange:
    def __init__(self, changes=None):
        if changes is None or len(changes) == 0:
            self.type = MapChangeType.NONE
            changes = []
        elif len(changes) == 1:
            self.type = MapChangeType.CHANGE
        elif len(changes) == 2:
            self.type = MapChangeType.MOVE_OR_INTERACT
        self.changes = changes

    def get_changes(self):
        return self.changes

    def get_type(self):
        return self.type


class ElementActionHandler:
    @staticmethod
    def apply(p: Point, action: LoderunnerAction, table, static_table):
        if action == LoderunnerAction.TICK:
            return MapChange()

        cur_elem = static_table[p.get_x()][p.get_y()]

        if action == action.SUICIDE:
            return ElementActionHandler.suicide_handler(p, table, static_table)


        elem_under_cur = table[p.get_x() + 1][p.get_y()]

        if cur_elem.get_name() == 'NONE' and elem_under_cur.get_name() in _ELEMENTS_CAN_FLIED:
            return MapChange([
                (p, cur_elem),
                # учитывать направление
                (Point(p.get_x() + 1, p.get_y()), Element('HERO_FALL_RIGHT'))
            ])

        if action == action.DO_NOTHING:
            return MapChange()

        if action == action.GO_RIGHT:
            return ElementActionHandler.go_forward_handler(p, table, static_table, "RIGHT")

        if action == action.GO_LEFT:
            return ElementActionHandler.go_forward_handler(p, table, static_table, "LEFT")

        if action == action.GO_DOWN:
            newp = Point(p.get_x() + 1, p.get_y())
            return ElementActionHandler.go_down_handler(p, newp, table, static_table)

        if action == action.GO_UP:
            newp = Point(p.get_x() - 1, p.get_y())
            return ElementActionHandler.go_up_handler(p, newp, table, static_table)

        if action == action.DRILL_LEFT:
            return ElementActionHandler.drill_handler(p, table, static_table, "LEFT")

        if action == action.DRILL_RIGHT:
            return ElementActionHandler.drill_handler(p, table, static_table, "RIGHT")
        
        if action == action.TICK:
            raise NotImplementedError()

        return MapChange()

    @staticmethod
    def drill_handler(p: Point, table, static_table, direction):
        sign = 1 if direction == "RIGHT" else -1
        target = table[p.get_x() + 1][p.get_y() + sign]

        if target.get_name() != 'BRICK':
            return MapChange()

        return MapChange([
            (p, Element('HERO_DRILL_' + direction)),
            (Point(p.get_x() + 1, p.get_y() + sign), Element('DRILL_PIT'))
        ])

    @staticmethod
    def suicide_handler(p: Point, table, static_table):
        result = [(p, static_table[p.get_x()][p.get_y()])]

        x = random.randrange(0, len(table))
        y = random.randrange(0, len(table))
        while (table[x][y].get_name() != 'NONE'):
            x = random.randrange(0, len(table))
            y = random.randrange(0, len(table))

        result.append((Point(x, y), Element('HERO_RIGHT')))
        return result

    @staticmethod
    def go_down_handler(p: Point, newp: Point, table, static_table):
        result = []
        cur_static_elem = static_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]
        if new_elem.get_name() == 'LADDER':
            result.append((p, cur_static_elem))
            result.append((newp, Element('HERO_LADDER')))

        if cur_static_elem.get_name() == 'PIPE':
            result.append((p, cur_static_elem))
            result.append((newp, Element('HERO_FALL_LEFT')))

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() in _ELEMENTS_CAN_FLIED:
            return MapChange([
                (p, cur_static_elem),
                (newp, Element('HERO_FALL_RIGHT'))
            ])

        return MapChange(result)

    @staticmethod
    def go_up_handler(p: Point, newp: Point, table, static_table):
        cur_static_elem = static_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'LADDER':
            return MapChange([
                (p, cur_static_elem),
                (newp, Element('HERO_LADDER'))
            ])

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'NONE':
            return MapChange([
                (p, cur_static_elem),
                (newp, Element('HERO_RIGHT'))
            ])

        return MapChange()

    @staticmethod
    def go_forward_handler(p: Point, table, static_table, direction):
        sign = 1 if direction == "RIGHT" else -1
        newp = Point(p.get_x(), p.get_y() + sign)
        cur_elem = static_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        if new_elem.get_char() in '#!☼':
            return MapChange()

        result = []
        if new_elem.get_char() in ' ~H$&@⊛S':
            result.append((p, cur_elem))
            if new_elem.get_name() == 'PIPE':
                result.append((newp, Element('HERO_PIPE_' + direction)))
            elif new_elem.get_name() == 'LADDER':
                result.append((newp, Element('HERO_LADDER')))
            elif new_elem.get_char() in ' $&@':
                result.append((newp, Element('HERO_' + direction)))
            elif new_elem.get_name() == 'PORTAL':
                result.append((newp, Element('HERO_' + direction)))
            elif new_elem.get_name() == 'THE_SHADOW_PILL':
                result.append((newp, Element('HERO_SHADOW_' + direction)))
        return MapChange(result)

    @staticmethod
    def is_valid_replacement(point, element, table):
        return True
