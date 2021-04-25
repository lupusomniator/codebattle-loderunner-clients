from loderunnerclient.util import count_perf
from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict
import numpy as np
import networkx as nx

from loderunnerclient.internals.element import Element
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.actions import LoderunnerAction

from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict

from loderunnerclient.graph.entities import is_entity, create_entity
from loderunnerclient.graph.space import is_space, is_available_space, create_space_element, \
    Direction, Brick, EmptySpace
from loderunnerclient.graph.actors import is_actor, create_actor
from loderunnerclient.graph.di_graph import \
    create_space_and_entry, \
    create_graph_no_edges, \
    fulfill_graph_edges_from_point, \
    get_ij, \
    NodeProps, \
    EdgeProps, \
    check_get_timer


class DynamicActionGraph:
    @count_perf
    def __init__(self, board, max_depth=20):
        self.initial_board = board
        self.max_depth = max_depth
        res = create_graph_no_edges(self.initial_board)
        self.graph: nx.DiGraph = res[0]
        self.hero_point: Point = res[1]
        self.timers = res[2]
        # add edges in graph around the hero in range of max_depth
        self.rebuild_graph_in_point(self.hero_point, max_depth=self.max_depth)
        
    @count_perf
    def rebuild_graph_in_point(self, p: Point, max_depth):
        return fulfill_graph_edges_from_point(self.graph, self.initial_board, p, max_depth, clone=False)

    @count_perf
    def tick(self, point_action: Tuple):
        # update graph
        self.update_graph_on_tick()

        # validate action
        if len(point_action) == 2:
            point, action = point_action
            actor = self.get_node_entry(point)
        elif len(point_action) == 3:
            # secondary-bot action
            point, action, actor = point_action
        else:
            assert False, f"shitty action {point_action}"

        is_secondary_bot_action = isinstance(actor, str)
        assert is_secondary_bot_action, f"For now this is for secondary bots only"

        found_edge = None
        for point_to in self.graph[point]:
            if action in self.graph[point][point_to][EdgeProps.actions]:
                found_edge = point, point_to
                break
        if found_edge is None:
            print(f"Unapplyable action {action} in point {point} of actor {actor}")
            return
        point_target = found_edge[1]

        # apply action
        if action in [LoderunnerAction.DRILL_RIGHT, LoderunnerAction.DRILL_LEFT]:
            self.get_node_space(point_target).set_drilling()
            timer = check_get_timer(self.get_node_space(point_target).element)
            assert timer is not None
            self.timers[point_target] = timer


    @count_perf
    def update_graph_on_tick(self):
        print("sd")
        exceeded_timers = self.update_timers()
        # for new pits remove edges to the node above
        # mark node and node above as unbuilt
        # run filling with edges from the node above
        for p in exceeded_timers["new_pits"]:
            up = p.coords() + Direction.up
            point_up = Point(*up)
            up_right = up + Direction.right
            point_up_right = Point(*up_right)
            up_left = up + Direction.left
            point_up_left = Point(*up_left)
            self.silent_remove_edge((point_up_left, point_up))
            self.silent_remove_edge((point_up_right, point_up))
            self.set_node_unbuild(p)
            self.set_node_unbuild(point_up)
            self.rebuild_graph_in_point(point_up, self.max_depth)
        # for new filling pits
        # run filling from node above
        for p in exceeded_timers["new_filling_pits"]:
            up = p.coords() + Direction.up
            point_up = Point(*up)
            self.set_node_unbuild(p)
            self.set_node_unbuild(point_up)
            self.rebuild_graph_in_point(point_up, self.max_depth)
        for p in exceeded_timers["new_filled_pits"]:
            up = p.coords() + Direction.up
            point_up = Point(*up)
            self.set_node_unbuild(p)
            self.set_node_unbuild(point_up)
            self.rebuild_graph_in_point(point_up, self.max_depth)

    @count_perf
    def update_timers(self):
        exceeded_timers = dict(
            new_pits=[],
            new_filling_pits=[],
            new_filled_pits=[]
        )
        timers = list(self.timers.items())
        for point, timer in timers:
            timer[1] -= 1
            timer_name = timer[0]
            if "PIT_FILL" in timer_name and timer[1] > 0:
                self.get_node_space(point).set_fill(timer[1])
            if timer[1] == 0:
                self.timers.pop(point)
                if timer_name == "DRILL_PIT":
                    element = Element("NONE")
                    self.set_node_space(point, create_space_element(element))
                    self.timers[point] = ("NONE", 1)
                    exceeded_timers["new_pits"].append(point)
                if timer_name == "NONE":
                    # TODO CHECK PIT_FILL SEQUENCE
                    element = Element("PIT_FILL_1")
                    self.set_node_space(point, create_space_element(element))
                    self.timers[point] = ("PIT_FILL", 4)
                    exceeded_timers['new_filling_pits'].append(point)
                if timer_name == "PIT_FILL":
                    element = Element("BRICK")
                    self.set_node_space(point, create_space_element(element))
                    exceeded_timers["new_filled_pits"].append(point)
        return exceeded_timers

    @count_perf
    def get_node_space(self, p: Point):
        return self.graph.nodes[p][NodeProps.space]

    @count_perf
    def set_node_space(self, p: Point, val):
        self.graph.nodes[p][NodeProps.space] = val

    @count_perf
    def get_node_entry(self, p: Point):
        return self.graph.nodes[p][NodeProps.entry]

    @count_perf
    def set_node_entry(self, p: Point, val):
        self.graph.nodes[p][NodeProps.entry] = val

    @count_perf
    def set_node_unbuild(self, p: Point):
        self.graph.nodes[p][NodeProps.build] = False

    @count_perf
    def get_node_is_build(self, p: Point):
        return self.graph.nodes[p][NodeProps.build]

    @count_perf
    def get_edge_actions(self, p1: Point, p2: Point):
        return self.graph.get_edge_data(p1, p2)[EdgeProps.actions]


    # # def get_action_transitions(self, p: Point):
    #     pass

    @count_perf
    def silent_remove_edge(self, node_tuple):
        try:
            self.graph.remove_edge(node_tuple)
        except nx.NetworkXError as e:
            pass

    def __getitem__(self, item):
        return self.graph[item]



def main():
    raise Exception("Cant be launched as main")

if __name__ == "__main__":
    main()
