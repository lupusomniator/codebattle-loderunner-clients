from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.element import Element, is_actor, to_falling, to_ladder, to_pipe, to_element
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
        else:
            assert False
        self.changes = changes

    def get_changes(self):
        return self.changes

    def get_type(self):
        return self.type

    def __str__(self):
        return str(self.changes)

    def __repr__(self):
        return str(self.changes)


class ElementActionHandler:
    @staticmethod
    def apply(p: Point, action: LoderunnerAction, table, static_table, cur_elem=None):
        if cur_elem is None:
            cur_elem = static_table[p.get_x()][p.get_y()]

        if action == action.SUICIDE:
            return ElementActionHandler.suicide_handler(p, table, static_table)

        elem_under_cur = table[p.get_x() + 1][p.get_y()]

        if action == LoderunnerAction.FILL_PIT:
            return ElementActionHandler.pit_fill_handler(p, table, static_table, cur_elem)

        if cur_elem.get_name() == 'NONE' and elem_under_cur.get_name() in _ELEMENTS_CAN_FLIED:
            return MapChange()

        if cur_elem.get_name() == 'NONE' and elem_under_cur.get_name() == 'PIPE':
            return MapChange()

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
    def pit_fill_handler(p: Point, table, static_table, element):
        if table[p.get_x()][p.get_y()].get_name() == "DRILL_PIT":
            return MapChange([(p, Element(" "))])

        if "PIT" in table[p.get_x()][p.get_y()].get_name():
            state = int(element.get_char())
            if state < 4:
                element = Element(str(state + 1))
            else:
                element = Element("BRICK")
        else:
            element = Element("PIT_FILL_1")

        return MapChange([(p, element)])

    @staticmethod
    def drill_handler(p: Point, table, static_table, direction):
        sign = 1 if direction == "RIGHT" else -1
        target = table[p.get_x() + 1][p.get_y() + sign]

        if target.get_name() != 'BRICK':
            return MapChange()

        if static_table[p.get_x()][p.get_y()].get_name() == 'PIPE':
            return MapChange()

        element_on_target = table[p.get_x()][p.get_y() + sign]
        if element_on_target.get_name() != 'NONE':
            return MapChange()

        prefix = ElementActionHandler.get_prefix(table[p.get_x()][p.get_y()])
        return MapChange([
            (p, Element(prefix + 'DRILL_' + direction)),
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

        result.append((Point(x, y), table[p.get_x()][p.get_y()]))
        assert len(result) < 3
        return MapChange(result)

    @staticmethod
    def go_down_handler(p: Point, newp: Point, table, static_table):
        result = []
        cur_static_elem = static_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        prefix = ElementActionHandler.get_prefix(table[p.get_x()][p.get_y()])
        if cur_static_elem.get_name() == 'PIPE':
            result.append((p, cur_static_elem))
            if new_elem.get_name() == 'LADDER':
                result.append((newp, Element(prefix + 'LADDER')))
            elif new_elem.get_name() == 'PIPE':
                result.append((newp, Element(prefix + 'PIPE_LEFT')))
            else:
                result.append((newp, Element(prefix + 'FALL_LEFT')))
        else:
            if new_elem.get_name() == 'LADDER':
                result.append((p, cur_static_elem))
                result.append((newp, to_ladder(table[p.get_x()][p.get_y()].get_name())))

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() in _ELEMENTS_CAN_FLIED:
            return MapChange([
                (p, cur_static_elem),
                (newp, to_falling(table[p.get_x()][p.get_y()].get_name()))
            ])
        assert len(result) < 3
        return MapChange(result)

    @staticmethod
    def go_up_handler(p: Point, newp: Point, table, static_table):
        cur_static_elem = static_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'LADDER':
            return MapChange([
                (p, cur_static_elem),
                (newp, to_ladder(table[p.get_x()][p.get_y()].get_name()))
            ])

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'NONE':
            return MapChange([
                (p, cur_static_elem),
                # fall from ladder
                (newp, table[p.get_x()][p.get_y()])
            ])

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'PIPE':
            return MapChange([
                (p, cur_static_elem),
                (newp, to_pipe(table[p.get_x()][p.get_y()].get_name()))
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

        prefix = ElementActionHandler.get_prefix(table[p.get_x()][p.get_y()])
        result = []
        if new_elem.get_char() in ' ~H$&@⊛S':
            result.append((p, cur_elem))
            if new_elem.get_name() == 'PIPE':
                result.append((newp, Element(prefix + 'PIPE_' + direction)))
            elif new_elem.get_name() == 'LADDER':
                result.append((newp, Element(prefix + 'LADDER')))
            elif new_elem.get_char() in ' $&@':
                result.append((newp, Element(prefix + direction)))
            elif new_elem.get_name() == 'PORTAL':
                result.append((newp, Element(prefix + direction)))
            elif new_elem.get_name() == 'THE_SHADOW_PILL':
                result.append((newp, Element(prefix + 'SHADOW_' + direction)))
        assert len(result) < 3
        return MapChange(result)

    @staticmethod
    def is_valid_replacement(point, element, table):
        old_element = table[point.get_x()][point.get_y()]
        if is_actor(element):
            if is_actor(old_element):
                return False
        return True

    @staticmethod
    def get_prefix(el):
        if 'OTHER_HERO' in el.get_name():
            return 'OTHER_HERO_'

        if 'HERO' in el.get_name():
            return 'HERO_'

        if 'ENEMY' in el.get_name():
            return 'ENEMY_'
