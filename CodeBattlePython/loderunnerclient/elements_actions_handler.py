from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.element import Element
from loderunnerclient.internals.constants import *


class ElementActionHandler:
    @staticmethod
    def apply(p: Point, action: LoderunnerAction, table, staitc_table):
        if action == LoderunnerAction.TICK:
            return []

        cur_elem = staitc_table[p.get_x()][p.get_y()]

        if action == action.SUICIDE:
            return [
                (p, cur_elem),
                #     появление в рандомном месте!
            ]

        elem_under_cur = table[p.get_x() + 1][p.get_y()]

        if cur_elem.get_name() == 'NONE' and elem_under_cur.get_name() in ELEMENTS_CAN_FLIED:
            return [
                (p, cur_elem),
                # учитывать направление
                (Point(p.get_x() + 1, p.get_y()), Element('HERO_FALL_RIGHT'))
            ]

        if action == action.DO_NOTHING:
            return []

        if action == action.GO_RIGHT:
            return ElementActionHandler.go_right_handler(p, table, staitc_table)

        if action == action.GO_LEFT:
            return ElementActionHandler.go_left_handler(p, table, staitc_table)

        if action == action.GO_DOWN:
            newp = Point(p.get_x() + 1, p.get_y())
            return ElementActionHandler.go_down_handler(p, newp, table, staitc_table)

        if action == action.GO_UP:
            newp = Point(p.get_x() - 1, p.get_y())
            return ElementActionHandler.go_up_handler(p, newp, table, staitc_table)

        return []

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

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() in ELEMENTS_CAN_FLIED:
            return [
                (p, cur_static_elem),
                (newp, Element('HERO_FALL_RIGHT'))
            ]

        return result

    @staticmethod
    def go_up_handler(p: Point, newp: Point, table, staitc_table):
        result = []
        cur_static_elem = staitc_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'LADDER':
            return [
                (p, cur_static_elem),
                (newp, Element('HERO_LADDER'))
            ]

        if cur_static_elem.get_name() == 'LADDER' and new_elem.get_name() == 'NONE':
            return [
                (p, cur_static_elem),
                (newp, Element('HERO_RIGHT'))
            ]

        return result

    @staticmethod
    def go_right_handler(p: Point, table, staitc_table):
        newp = Point(p.get_x(), p.get_y() + 1)
        cur_elem = staitc_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        if new_elem.get_char() in '#!☼':
            return []

        result = []
        if new_elem.get_char() in ' ~H$&@⊛S':
            result.append((p, cur_elem))
            if new_elem.get_name() == 'PIPE':
                result.append((newp, Element('HERO_PIPE_RIGHT')))
                return result

            if new_elem.get_name() == 'LADDER':
                result.append((newp, Element('HERO_LADDER')))
                return result

            if new_elem.get_char() in ' $&@':
                result.append((newp, Element('HERO_RIGHT')))
                return result

            # телепортировать чувака
            if new_elem.get_name() == 'PORTAL':
                result.append((newp, Element('HERO_RIGHT')))
                return result

            if new_elem.get_name() == 'THE_SHADOW_PILL':
                result.append((newp, Element('HERO_SHADOW_RIGHT')))
                return result
        return result

    @staticmethod
    def go_left_handler(p: Point, table, staitc_table):
        newp = Point(p.get_x(), p.get_y() - 1)
        cur_elem = staitc_table[p.get_x()][p.get_y()]
        new_elem = table[newp.get_x()][newp.get_y()]

        print(new_elem)
        if new_elem.get_char() in '#!☼':
            return []

        result = []
        if new_elem.get_char() in ' ~H$&@⊛S':
            result.append((p, cur_elem))
            if new_elem.get_name() == 'PIPE':
                result.append((newp, Element('HERO_PIPE_LEFT')))
                return result

            if new_elem.get_name() == 'LADDER':
                result.append((newp, Element('HERO_LADDER')))
                return result

            if new_elem.get_char() in ' $&@':
                result.append((newp, Element('HERO_LEFT')))
                return result

            # телепортировать чувака
            if new_elem.get_name() == 'PORTAL':
                result.append((newp, Element('HERO_LEFT')))
                return result

            if new_elem.get_name() == 'THE_SHADOW_PILL':
                result.append((newp, Element('HERO_SHADOW_LEFT')))
                return result
        return result

    @staticmethod
    def is_valid_replacement(point, element, table):
        return True
