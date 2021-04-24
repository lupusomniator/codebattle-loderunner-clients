from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict
import numpy as np
import networkx as nx

from loderunnerclient.internals.element import Element
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.point import Point
from loderunnerclient.internals.actions import LoderunnerAction

from loderunnerclient.graph.entities import is_entity, create_entity
from loderunnerclient.graph.space import is_space, is_available_space, create_space_element, \
    Direction
from loderunnerclient.graph.actors import is_actor, create_actor

from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict
import numpy as np
import networkx as nx

from loderunnerclient.internals.element import Element
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.point import Point

from loderunnerclient.graph.entities import is_entity, create_entity
from loderunnerclient.graph.space import is_space, is_available_space, create_space_element, \
    Direction
from loderunnerclient.graph.actors import is_actor, create_actor
from loderunnerclient.graph.multi_di_graph import \
    create_space_and_entry, \
    create_graph_no_edges, \
    fulfill_graph_edges_from_point, \
    get_ij


class DynamicActionGraph:
    def __init__(self, board):
        self.initial_board = board
        self.graph, self.hero_point = create_graph_no_edges(self.initial_board)
        # add edges in graph around the hero in range of max_depth
        self.graph = fulfill_graph_edges_from_point(self.graph, self.initial_board,
                                                    self.hero_point, max_depth=20)
        self.timers = dict()

    def tick(self, point_action: Tuple[Point, LoderunnerAction]):
        # TODO check if action is applyable
        # TODO apply action
        # TODO ------optionally rebuild graph
        # TODO update timers ticks
        #       TODO rebuild graph in points where timer is over
        pass



def main():
    raise Exception("Cant be launched as main")

if __name__ == "__main__":
    main()
