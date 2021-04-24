from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict
import numpy as np
import networkx as nx

from loderunnerclient.internals.element import Element
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.actions import LoderunnerAction
from loderunnerclient.internals.point import Point

from loderunnerclient.graph.entities import is_entity, create_entity
from loderunnerclient.graph.space import is_space, is_available_space, create_space_element, Direction
from loderunnerclient.graph.actors import is_actor, create_actor
from loderunnerclient.graph.di_graph import create_graph_no_edges, fulfill_graph_edges_from_point


class Ant:
    def __init__(self, p: Point, graph: nx.DiGraph, board: Board,
                 game_emulator=None, initial_transition: Point=None):
        self.start_point: Point = p
        self.graph: nx.DiGraph = graph
        self.board: Board = board
        self.game_emulator = game_emulator
        self.initial_transition: Point = initial_transition
        self.history: List = []
        self.path = List[Point] = []

    #  идти по графу случайным образом на фиксированную глубину (не делая шаг назад)
    #  возвращает список встреченных сущностей
    def walk(self, max_depth):
        graph = self.graph
        history = self.history
        path = self.path

        depth = 0
        cur_point = self.start_point
        prev_point = None
        if self.initial_transition:
            depth = 1
            cur_point = self.initial_transition
            prev_point = self.start_point

        while depth <= max_depth:
            if graph[cur_point]["builded"] is False:
                fulfill_graph_edges_from_point(self.graph, self.board, cur_point, max_depth+1)
            if cur_point != self.start_point:
                history.append(graph.nodes[cur_point]["entry"])  # None or Entity or Actor
                path.append(cur_point)
            available_pts_to_move = list(set(graph[cur_point].keys()) - set([prev_point]))
            rand_ind = np.random.randint(len(available_pts_to_move))
            chosen_pt_to_move = available_pts_to_move[rand_ind]
            # make step
            depth += 1
            prev_point = cur_point
            cur_point = chosen_pt_to_move





class GreedyAntBot:
    def __init__(self):
        self.board = None
        self.graph = None

    def make_action(self, board: Board) -> LoderunnerAction:
        pass




