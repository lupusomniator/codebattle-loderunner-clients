from loderunnerclient.util import count_perf, PerfStat
from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict
import numpy as np
import networkx as nx
from copy import copy, deepcopy
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
    @count_perf
    def __init__(self, p: Point, graph: DynamicActionGraph, max_depth,
                 game_emulator=None, initial_transition: Tuple[Point, LoderunnerAction] = None):
        self.start_point: Point = p
        self.graph: DynamicActionGraph = copy(graph)
        # graph.graph = nx.DiGraph(graph.graph)
        self.max_depth = max_depth
        self.game_emulator = game_emulator
        self.initial_transition: Tuple[Point, LoderunnerAction] = initial_transition

        self.history: List = [None] * self.max_depth
        self.path: List[Point] = [None] * self.max_depth
        self.action_sequence: List[LoderunnerAction] = [None] * self.max_depth  # actions (graph edges)

        self.reward = 0

    #  идти по графу случайным образом на фиксированную глубину (не делая шаг назад)
    #  возвращает список встреченных сущностей
    @count_perf
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
            print(cur_point, graph.get_node_entry(cur_point))
            if graph.get_node_is_build(cur_point) is False:
                graph.rebuild_graph_in_point(cur_point, max_depth // 2)
            if cur_point != self.start_point:
                cur_entry = graph.get_node_entry(cur_point)
                history[depth - 1] = cur_entry  # None or Entity or Actor
                if not cur_entry is None:
                    self.reward += cur_entry.get_reward()
                path[depth - 1] = cur_point
                action_sequence[depth - 1] = prev_action
                if cur_entry == "PORTAL":
                    break
                if is_dangerous_actor(cur_entry) or is_not_dangerous_actor(cur_entry):
                    break

            #  allowed to go everywhere except going back and going to itself (do_nothing or suicide)
            available_pts_to_move = list(set(graph.graph[cur_point].keys()) - set([prev_point, cur_point]))
            if len(available_pts_to_move) == 0:
                #  jumped in the gap or something
                break
            chosen_random_pt_to_move = available_pts_to_move[np.random.randint(len(available_pts_to_move))]

            # all edges has only one action (only loop to itself contains 2 possible actions)

            chosen_action = graph.get_edge_actions(cur_point, chosen_random_pt_to_move)[0]
            if chosen_action is None:
                break

            # make step
            depth += 1
            prev_point = cur_point
            cur_point = chosen_random_pt_to_move
            prev_action = chosen_action

        return self


class GreedyAntBot(AbstractBot):
    def __init__(self, ant_count=20, max_depth=20, print_stat=True):
        super().__init__()
        self.board = None
        self.graph = None
        self.ant_count = ant_count
        self.max_depth = max_depth
    
    @count_perf
    def choose_action(self, board: Board) -> LoderunnerAction:
        PerfStat.print_stat()
        # return LoderunnerAction.SUICIDE
        hero_pos = board.get_my_position()

        dag = DynamicActionGraph(board, self.max_depth)

        ants = []
        for i in range(self.ant_count):
            ants.append(Ant(hero_pos, dag, self.max_depth).walk())
            print(ants[-1].action_sequence)
            print(ants[-1].reward)
        rewards = [ant.reward for ant in ants]

        best_ant = ants[rewards.index(max(rewards))]

        # если лучший никуда не смог пойти
        if best_ant.action_sequence[0] is None:
            return LoderunnerAction.SUICIDE

        return best_ant.action_sequence[0]
