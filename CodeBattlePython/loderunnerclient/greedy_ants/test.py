import networkx as nx
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.element import _ELEMENTS, Element, value_of#, index_to_char, char_to_index

G = nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3, attr=dict(KEY="VAL"))
G.add_edge(1,2, transition="teleport")
G.add_edge(1,3, transition="biba")
G.add_edge(2,1, transition="normal")
G.add_edge(2,1, trans="NEW")

print(G.edges())
print(G.get_edge_data(2,1))
print(G.nodes[3])
print(G[1][3])
# print(G[1]["entry"])

print("======================")

G = nx.MultiDiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3, attr=dict(KEY="VAL"))
G.add_edge(1,2, transition="teleport")
G.add_edge(1,3, transition="biba")
print("G has 2,1", G.has_edge(2,1))
G.add_edge(2,1, transition="normal")
print("G has 2,1", G.has_edge(2,1))
G.add_edge(2,1, trans="NEW")
print("G has 2,1", G.has_edge(2,1))

print(G.edges())
print(G.get_edge_data(2,1))

print(G.nodes[3])
print(G[1][3])