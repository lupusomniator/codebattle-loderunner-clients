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
from loderunnerclient.graph.actors import is_actor, create_actor, is_dangerous_actor, is_not_dangerous_actor
from loderunnerclient.graph.di_graph import create_graph_no_edges, fulfill_graph_edges_from_point
from loderunnerclient.graph.dynamic_action_graph import DynamicActionGraph
from loderunnerclient.bots.abstract_bot import AbstractBot


class Ant:
    def __init__(self, p: Point, graph: DynamicActionGraph, max_depth,
                 game_emulator=None, initial_transition: Tuple[Point, LoderunnerAction]=None):
        self.start_point: Point = p
        self.graph: DynamicActionGraph = graph
        self.max_depth = max_depth
        self.game_emulator = game_emulator
        self.initial_transition: Tuple[Point, LoderunnerAction] = initial_transition

        self.history: List = [None] * self.max_depth
        self.path: List[Point] = [None] * self.max_depth
        self.action_sequence: List[LoderunnerAction] = [None] * self.max_depth  # actions (graph edges)

    #  идти по графу случайным образом на фиксированную глубину (не делая шаг назад)
    #  возвращает список встреченных сущностей
    def walk(self):
        graph = self.graph
        history = self.history
        path = self.path
        action_sequence = self.action_sequence
        max_depth = self.max_depth

        depth = 0
        cur_point = self.start_point
        prev_point = None
        prev_action = None
        if self.initial_transition:
            initial_point_to, initial_action = self.initial_transition
            depth = 1
            cur_point = initial_point_to
            prev_point = self.start_point
            prev_action = initial_action

        while depth <= max_depth:
            if graph.get_node_is_build(cur_point) is False:
                graph.rebuild_graph_in_point(cur_point, max_depth // 2)
            if cur_point != self.start_point:
                cur_entry = graph.get_node_entry(cur_point)
                if is_dangerous_actor(cur_entry) or is_not_dangerous_actor(cur_entry):
                    break
                history[depth - 1] = cur_entry  # None or Entity or Actor
                path[depth - 1] = cur_point
                action_sequence[depth - 1] = prev_action

            #  allowed to go everywhere except going back and going to itself (do_nothing or suicide)
            available_pts_to_move = list(set(graph.graph[cur_point].keys()) - set([prev_point, cur_point]))
            if len(available_pts_to_move) == 0:
                #  jumped in the gap or something
                break
            chosen_random_pt_to_move = available_pts_to_move[np.random.randint(len(available_pts_to_move))]
            # all edges has only one action (only loop to itself contains 2 possible actions)
            chosen_action = graph.get_edge_actions((prev_point, chosen_random_pt_to_move))[0]

            # make step
            depth += 1
            prev_point = cur_point
            cur_point = chosen_random_pt_to_move
            prev_action = chosen_action


class GreedyAntBot(AbstractBot):
    def __init__(self):
        super().__init__()
        self.board = None
        self.graph = None

    def choose_action(self, board: Board) -> LoderunnerAction:
        pass




