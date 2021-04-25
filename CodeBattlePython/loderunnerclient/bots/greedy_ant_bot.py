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
from loderunnerclient.graph.space import is_space, is_available_space, create_space_element, Direction, EmptySpace
from loderunnerclient.graph.actors import is_actor, create_actor, is_dangerous_actor, is_not_dangerous_actor
from loderunnerclient.graph.di_graph import create_graph_no_edges, fulfill_graph_edges_from_point, add_move_edges
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
        self.action_sequence: List[LoderunnerAction] = [
                                                           None] * self.max_depth  # actions (graph edges)

        self.reward = 0

    def get_attenuation(self, step_num):
        return ((self.max_depth - step_num + 1) / self.max_depth) ** 2

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

        edges_to_remove_back = defaultdict(list)  # key = depth (tick) when to remove edges, val = list of edges
        edges_to_add_back = defaultdict(list)  # key = depth (tick) when to add edges, val = list of edges
        while depth <= max_depth:
            for p1, p2 in edges_to_remove_back[depth]:
                self.graph.graph.remove_edge(p1,p2)
            for p1, p2 in edges_to_add_back[depth]:
                self.graph.graph.add_edge(p1,p2)
            if graph.get_node_is_build(cur_point) is False:
                graph.rebuild_graph_in_point(cur_point, max_depth // 2)
            if cur_point != self.start_point:
                cur_entry = graph.get_node_entry(cur_point)
                history[depth - 1] = cur_entry  # None or Entity or Actor
                if not cur_entry is None:
                    self.reward += cur_entry.get_reward() * self.get_attenuation(depth)

                path[depth - 1] = cur_point
                action_sequence[depth - 1] = prev_action
                if cur_entry == "PORTAL":
                    break
                if is_dangerous_actor(cur_entry) or is_not_dangerous_actor(cur_entry):
                    break

            #  allowed to go everywhere except going back and going to itself (do_nothing or suicide)
            available_pts_to_move = list(
                set(graph.graph[cur_point].keys()) - set([prev_point, cur_point]))
            if len(available_pts_to_move) == 0:
                #  jumped in the gap or something
                break
            chosen_random_pt_to_move = available_pts_to_move[
                np.random.randint(len(available_pts_to_move))]

            # all edges has only one action (only loop to itself contains 2 possible actions)

            chosen_action = graph.get_edge_actions(cur_point, chosen_random_pt_to_move)[0]

            # make step
            if chosen_action in [LoderunnerAction.DRILL_LEFT, LoderunnerAction.DRILL_RIGHT]:
                # temporary add edges in graoh
                drilling_point = chosen_random_pt_to_move
                point_above_drilling = Point(*(drilling_point.coords() + Direction.up))
                pit_life_length = 3
                time_when_to_turn_back = depth + pit_life_length
                edges_to_remove_back[time_when_to_turn_back].extend(
                    add_move_edges(self.graph.graph, self.graph.initial_board,
                                   drilling_point, False,
                                   mock_space=EmptySpace(Element("NONE"))))
                temp_added_edges = [(point_above_drilling, Point(0,1) + point_above_drilling)]
                for p1,p2 in temp_added_edges:
                    self.graph.graph.add_edge(p1,p2)
                edges_to_remove_back[time_when_to_turn_back].extend(temp_added_edges)

                temp_removed_edges = [(Point(-1, 0) + point_above_drilling, point_above_drilling),
                                      (Point(1, 0) + point_above_drilling, point_above_drilling),
                                      (Point(-1, 0) + point_above_drilling, drilling_point),
                                      (Point(1, 0) + point_above_drilling, drilling_point)]
                for p1,p2 in temp_removed_edges:
                    self.graph.graph.remove_edge(p1,p2)
                edges_to_add_back[time_when_to_turn_back].extend(temp_removed_edges)


            if chosen_action in [LoderunnerAction.GO_LEFT, LoderunnerAction.GO_RIGHT, LoderunnerAction.GO_DOWN, LoderunnerAction.GO_UP]:
                prev_point = cur_point
                cur_point = chosen_random_pt_to_move

            prev_action = chosen_action
            depth += 1

        for i, a in enumerate(action_sequence):
            if a is None:
                self.reward -= 0.1 * self.get_attenuation(i)

        return self


class GreedyAntBot(AbstractBot):
    def __init__(self, ant_count=20, max_depth=20, print_stat=True):
        super().__init__()
        self.board = None
        self.graph = None
        self.ant_count = ant_count
        self.max_depth = max_depth
        self.print_stat = print_stat
        self.last_pos = None
    
    @count_perf
    def choose_action(self, board: Board) -> LoderunnerAction:
        # PerfStat.print_stat()
        # return LoderunnerAction.SUICIDE
        hero_pos = board.get_my_position()
        if self.last_pos is None:
            self.last_pos = hero_pos

        dag = DynamicActionGraph(board, self.max_depth)
    
        # get edges from current point
        edges = dag[hero_pos]
        # points where edges are going
        points_to = list(set(edges.keys()) - set([hero_pos]))
        if len(points_to) == 0:
            return LoderunnerAction.SUICIDE

        max_action = None
        max_reward = -10000000
        for point in points_to:
            actions = dag.get_edge_actions(hero_pos, point)
            total_reward = 0
            if not actions is None:
                action = actions[0]
                for i in range(self.ant_count):
                    ant = Ant(hero_pos, dag, self.max_depth, initial_transition=(point, action))
                    total_reward += ant.walk().reward
                print(point, action, total_reward)
                if total_reward > max_reward:
                    max_action = action
                    max_reward = total_reward
        if max_action is None:
            return LoderunnerAction.SUICIDE
        print("MAX ACTION:", max_action, max_reward)
        return max_action
