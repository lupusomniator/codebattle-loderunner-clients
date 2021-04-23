import random
import time
from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.element import is_actor, is_holding_actor
from loderunnerclient.elements_actions_handler import ElementActionHandler
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.constants import *

def print_table(table):
    for line in table:
        for element in line:
            print(element.get_char(), end="")
        print()

def ask_for_next_action():
    print("input next action:", end="")
    user_input = input()
    if user_input == "w":
        return LoderunnerAction.GO_UP
    if user_input == "s":
        return LoderunnerAction.GO_DOWN
    if user_input == "a":
        return LoderunnerAction.GO_LEFT
    if user_input == "d":
        return LoderunnerAction.GO_RIGHT
    if user_input == "q":
        return LoderunnerAction.DRILL_LEFT
    if user_input == "e":
        return LoderunnerAction.DRILL_RIGHT
    assert False

class Game:
    def __init__(self, board):
        self.mutable_board = board.get_elements_table(static_only=False)
        self.static_board = board.get_elements_table(static_only=True)
        self.brick_positions = board.get_brick_positions()      
        print_table(self.mutable_board)

    def apply_actions(self, actions_list):
        """
        Применяет действия из списка к текущей доске (порядок имеет значение!)

        Parameters
        ----------
        actions_list : list of tuples
            Список пар: координата точки и действие, которое должен применить объект стоящий в ней

        Returns
        -------
        Возвращает список элементов и их координат

        """
        new_elements_list = []
        for point, action in actions_list:
            new_elements_list.extend(ElementActionHandler.apply(
                point,
                action,
                self.mutable_board,
                self.static_board
            ))
        return new_elements_list

    def update_world(self, elements_list):
        """
        Изменяет текущее состояние доски в соответствии со списком

        Parameters
        ----------
        elements_list : list of tuples
            Список пар: координата точки и объект Element, который теперь будет содержаться в этой точке
        """
        for point, element in elements_list:
            is_valid = ElementActionHandler.is_valid_replacement(
                point,
                element,
                self.mutable_board
            )
            if is_valid:
                x, y = point.get_x(), point.get_y()
                old_element = self.mutable_board[x][y]
                self.mutable_board[x][y] = element
                # TODO: if old_elemet == gold and element == player then increase player gold
                # and so on
    
    def get_world_actions(self):
        """
        Действия, которые должен совершить мир и на которые пользователи не могут повлиять

        Returns
        -------
        new_elements_list : TYPE
            DESCRIPTION.

        """
        new_elements_list = []
        # Зарастание ям
        for point in self.brick_positions:
            new_elements_list.extend(ElementActionHandler.apply(
                point,
                LoderunnerAction.TICK,
                self.mutable_board,
                self.static_board
            ))

        # Падение героев и врагов
        not_holding_actors = self.find_all_not_holding_actors()
        for point, element in not_holding_actors:
            x, y = point.get_x(), point.get_y()
            down_element = self.mutable_board[x + 1][y]
            if down_element.get_name() in ELEMENTS_CAN_FLIED:
                new_elements_list.extend([
                    (Point(x, y), self.static_board[x + 1][y]),
                    (Point(x + 1, y), element)
                ])
        # TODO: enemy stategy
        return new_elements_list

    def do_tick(self, users_actions):
        """
        Выполняет один тик в соответствии с действиями пользователей и меняет мир
        """
        replacement_queue = self.apply_actions(users_actions)
        replacement_queue.extend(self.get_world_actions())
        self.update_world(replacement_queue)

    def get_random_users_actions(self):
        users_points = []
        for i, line in enumerate(self.mutable_board):
            for j, element in enumerate(line):
                if element.get_name().startswith("HERO") or element.get_name().startswith("OTHER_HERO"):
                    users_points.append(Point(i, j))
        return [
            (point, list(LoderunnerAction)[random.randint(0, len(LoderunnerAction) - 3)])
            for point in users_points
        ]

    def find_all_heroes(self):
        heroes = []
        for i, line in enumerate(self.mutable_board):
            for j, element in enumerate(line):
                if element.get_name().startswith("HERO") or element.get_name().startswith("OTHER_HERO"):
                    heroes.append((Point(i, j), element))
        return heroes    

    def find_all_holding_actors(self):
        heroes = []
        for i, line in enumerate(self.mutable_board):
            for j, element in enumerate(line):
                if is_holding_actor(element):
                    heroes.append((Point(i, j), element))
        return heroes    

    def find_all_not_holding_actors(self):
        heroes = []
        for i, line in enumerate(self.mutable_board):
            for j, element in enumerate(line):
                if is_actor(element) and not is_holding_actor(element):
                    heroes.append((Point(i, j), element))
        return heroes

    def find_hero(self):
        for i, line in enumerate(self.mutable_board):
            for j, element in enumerate(line):
                if element.get_name().startswith("HERO"):
                    return (Point(i, j), element)
        

    def run(self, ticks=100, render=False):
        """
        Запускает игру
        
        Parameters
        ----------
        ticks : Количество тиков до завершения игры

        Returns
        -------
        None.

        """
        for i in range(ticks):
            self.do_tick([(self.find_hero()[0], ask_for_next_action())])
            if render:
                print_table(self.mutable_board)
                print("=" * 30)

if __name__ == "__main__":
    board = Board.load_from_file("last_board")
    game = Game(Board.load_from_file("last_board"))
    game.run(render=True)