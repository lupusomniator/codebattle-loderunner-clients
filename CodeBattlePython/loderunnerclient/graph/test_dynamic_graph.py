from loderunnerclient.graph.dynamic_action_graph import DynamicActionGraph
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.point import Point

from loderunnerclient.graph.di_graph import \
    create_space_and_entry, \
    create_graph_no_edges, \
    fulfill_graph_edges_from_point, \
    get_ij


board_str = open("./resource/board_example", "r").read()
board = Board(board_str)

# g = DynamicActionGraph(board)


graph, hero_point, timers = create_graph_no_edges(board)
p = Point(19, 12)
# add edges in graph around the hero in range of max_depth
graph, met_objects, report = fulfill_graph_edges_from_point(graph, board,
                                            p, max_depth=21, verbose=True)

p_check = p
print(f"edges for {p_check}")
[print(k,v) for k,v in graph[p_check].items()]
print(report)