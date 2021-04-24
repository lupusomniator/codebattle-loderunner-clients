import random
import time
from collections import defaultdict
from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.element import Element, is_actor, is_holding_actor
from loderunnerclient.elements_actions_handler import ElementActionHandler, MapChangeType, MapChange
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.constants import _ELEMENTS_CAN_FLIED

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
    if user_input == "z":
        return LoderunnerAction.SUICIDE
    assert False

class RewardType:
    GREEN=("GREEN", 1),
    YELLOW=("YELLOW", 2),
    RED=("RED", 3),
    KILL=("KILL", 10),
    DIE=("DIE", -1)

class Score:
    def __init__(self):
        self.__reset_score__()

    def __reset_score__(self):
        self.gold_strick = {
            "GREEN": 0,
            "YELLOW": 0,
            "RED": 0
        }
        self.score = 0

    def reward(self, rewardType):
        if (rewardType == RewardType.GREEN or 
            rewardType == RewardType.YELLOW or
            rewardType == RewardType.RED
        ):
            self.gold_strick[rewardType[0]] += 1
            self.score += self.gold_strick[rewardType[0]] * rewardType[1]
        if rewardType == RewardType.KILL:
            self.score += 10
        if rewardType == RewardType.DIE:
            self.__reset_score__()

class Brick:
    TICKS_TILL_FILL = 3

    def __init__(self, element, counter = 0):
        self.element = element
        self.counter = counter

    def tick(self):
        self.counter += 1
        if self.element.get_name() == "PIT_FILL_4":
            self.element = Element("BRICK")

        if self.element.get_name() == "BRICK":
            self.counter = 0
            return False

        return True
        # MOVE TO EVENTS HANDLER
        # if self.counter == Brick.TICKS_TILL_FILL:
        #     self.element = Element("PIT_FILL_1")
        # elif self.counter > Brick.TICKS_TILL_FILL:
        #     self.element = Element(str(int(self.element.get_char()) + 1))

        # return MapAction([self.point, self.element])

class Player:
    TICKS_TILL_SHADOW_EFFECT_ENDS = 5

    def __init__(self, element, score=Score()):
        self.element = element
        self.score = score

    def reward(self, rewardType):
        self.score.reward(rewardType)

    def update_element(self, element):
        self.element = element

class Game:
    def add_to_table(self, table, point, value):
        table[self.cur_max_index] = value
        self.cur_max_index += 1        
        
    def update_player_position(self, old_point, new_point):
        value = self.players_table[old_point]
        assert new_point not in self.players_table
        self.players_table.pop(old_point)
        self.players_table[new_point] = value

    def update_enemy_position(self, old_point, new_point):
        assert old_point in self.enemies_positions
        self.enemies_positions[old_point] -= 1
        self.enemies_positions[new_point] += 1
        
    def __init__(self, board):
        self.cur_max_index = 0

        self.mutable_board = board.get_elements_board(static_only=False)
        self.static_board = board.get_elements_board(static_only=True)

        self.bricks_table = dict()
        self.enemies_table = defaultdict(int)
        self.players_table = dict()
        self.indexes_table = dict()

        brick_positions = set(board.get_brick_positions())
        for point in brick_positions:
            self.add_to_table(
                self.bricks_table,
                point,
                Brick(self.static_board[point.get_x()][point.get_y()])
            )

        enemies_positions = set(board.get_enemy_positions())
        for point in enemies_positions:
            self.add_to_table(
                self.enemies_table, 
                point,
                self.enemies_table[point] + 1
            )

        heroes_positions = {board.get_my_position()}
        heroes_positions.update(board.get_other_hero_positions())
        for point in enemies_positions:
            self.add_to_table(
                self.players_table, 
                point,
                Player(self.mutable_board[point.get_x()][point.get_y()])
            )

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
        new_changes_list = []
        for point, action in actions_list:
            new_changes_list.append(ElementActionHandler.apply(
                point,
                action,
                self.mutable_board,
                self.static_board
            ))
        return new_changes_list

    def update_world(self, changes_list):
        """
        Изменяет текущее состояние доски в соответствии со списком

        Parameters
        ----------
        changes_list : list of MapChange
            Список пар: координата точки и объект Element, который теперь будет содержаться в этой точке
        """
        def is_valid(map_change):
            if map_change.get_type() == MapChangeType.NONE:
                return True
            point, element = map_change.get_changes()[-1]
            return ElementActionHandler.is_valid_replacement(
                point,
                element,
                self.mutable_board
            )

        def apply_change(point, element):
            x, y = point.get_x(), point.get_y()
            self.mutable_board[x][y] = element

        for change in changes_list:
            if is_valid(change):
                if change.get_type() == MapChangeType.NONE:
                    continue
                elif change.get_type() == MapChangeType.CHANGE:
                    # TODO: update tables
                    apply_change(*change.get_changes()[-1])
                elif change.get_type() == MapChangeType.MOVE:
                    # TODO: update tables
                    src, dst = change.get_changes()
                    apply_change(*src)
                    apply_change(*dst)
    
    def get_world_actions(self):
        """
        Действия, которые должен совершить мир и на которые пользователи не могут повлиять

        Returns
        -------
        new_changes_list : TYPE
            DESCRIPTION.

        """
        new_changes_list = []
        # Зарастание ям
        # for point in self.brick_positions:
        #     new_changes_list.append(ElementActionHandler.apply(
        #         point,
        #         LoderunnerAction.TICK,
        #         self.mutable_board,
        #         self.static_board
        #     ))

        # Падение героев и врагов
        not_holding_actors = self.find_all_not_holding_actors()
        for point, element in not_holding_actors:
            x, y = point.get_x(), point.get_y()
            down_element = self.mutable_board[x + 1][y]
            if down_element.get_name() in _ELEMENTS_CAN_FLIED:
                # TODO: LoderunnerAction.FALL
                new_changes_list.append(MapChange([
                    (Point(x, y), self.static_board[x + 1][y]),
                    (Point(x + 1, y), element)
                ]))
        # TODO: enemy stategy
        return new_changes_list

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