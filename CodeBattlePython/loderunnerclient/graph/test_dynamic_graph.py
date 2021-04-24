from .dynamic_action_graph import DynamicActionGraph
from loderunnerclient.internals.board import Board

board_str = open("./resource/board_example", "r").read()
board = Board(board_str)

g = DynamicActionGraph(board)