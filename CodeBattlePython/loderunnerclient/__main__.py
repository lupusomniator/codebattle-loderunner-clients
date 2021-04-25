import logging
from loderunnerclient.game_client import GameClient
# from loderunnerclient.environment import Environment
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.game import Game
from loderunnerclient.graph.dynamic_action_graph import DynamicActionGraph
from loderunnerclient.bots.greedy_ant_bot import Ant
import numpy as np
from loderunnerclient.util import PerfStat


logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

def turn(board):
    board.save_to_file("last_board")

def local_main():
    board = Board.load_from_file("last_board")
    dag = DynamicActionGraph(board, 20)
    
    my_pos = board.get_my_position()
    for i in range(1000):
        ant = Ant(my_pos, dag, 20)
        ant.walk()
    print(ant.action_sequence)

def server_main():
    # env = Environment()
    gcb = GameClient(
        # change this url to your
        "https://dojorena.io/codenjoy-contest/board/player/dojorena392?code=407418550408423703"
    )
    # gcb.run(env.on_turn)


if __name__ == "__main__":
    # Board.load_from_file("last_board")
    # for i in range(10000):
        # turn(Board.load_from_file("last_board"))
    #main()
    # server_main()
    local_main()
    PerfStat.print_stat()
