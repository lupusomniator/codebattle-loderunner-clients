from typing import Iterable, Union, Optional, Tuple, Any, List, Dict, DefaultDict, Set
from collections import defaultdict
import numpy as np
import networkx as nx

from loderunnerclient.internals.element import Element
from loderunnerclient.internals.board import Board
from loderunnerclient.internals.point import Point

from loderunnerclient.graph.entities import is_entity, create_entity
from loderunnerclient.graph.space import is_space, is_available_space, create_space_element, Direction
from loderunnerclient.graph.actors import is_actor, create_actor


def create_space_and_entry(element: Element) -> Tuple[Any, Any]:
    if is_space(element):
        return create_space_element(element), None
    if is_entity(element):
        return create_space_element(" "), create_entity(element)
    if is_actor(element):
        actor = create_actor(element)
        if actor.is_on_ladder:
            return create_space_element("LADDER"), actor
        elif actor.is_on_pipe:
            return create_space_element("PIPE"), actor
        else:
            return create_space_element("NONE"), actor


def create_graph_no_edges(board: Board):
    graph = nx.DiGraph()
    start_point = None
    for strpos, char in enumerate(board._string):
        point = board._strpos2pt(strpos)
        element = Element(char)
        space, entry = create_space_and_entry(element)
        graph.add_node(point, space=space, entry=entry, builded=False)
        if entry and entry.name == "HERO":
            start_point = point
    return graph, start_point


def get_ij(point):
    point_coords = [point.get_y() + 1, point.get_x() + 1]
    return point_coords


def fulfill_graph_edges_from_point(graph: nx.DiGraph, board: Board, start_point: Point,
                                   max_depth: int = 10, verbose=False):
    graph = nx.DiGraph(graph)
    closest_distances: DefaultDict[Point] = defaultdict(lambda: float('inf'))
    closest_distances[start_point] = 0

    met_objects: DefaultDict[str] = defaultdict(set)

    start_task = (start_point, 0)
    task_stack: List[Tuple[Point, int]] = [start_task]

    iter = 0
    while len(task_stack):
        iter += 1
        task = task_stack.pop()
        cur_point, depth = task
        new_depth = depth + 1
        if new_depth > max_depth:
            continue

        cur_space = graph.nodes[cur_point]["space"]
        cur_entry = graph.nodes[cur_point]["entry"]
        cur_coords = np.array((cur_point._x, cur_point._y))

        if graph.nodes[cur_point]["builded"] and closest_distances[cur_point] <= depth:
            continue
        if cur_space.available_moves.size == 0:
            continue

        if closest_distances[cur_point] > depth:
            closest_distances[cur_point] = depth

        if cur_entry:
            met_objects[cur_entry.name].add(cur_point)
        graph.nodes[cur_point]["builded"] = True

        space_down = graph.nodes[Point(*(cur_coords + Direction.down))]["space"]
        is_on_air = space_down.name == "NONE"

        if verbose:
            print(f"Chosen moves for cur_point {get_ij(cur_point)}:")
            print("space", cur_space.name)
            print("entry", cur_entry.name if cur_entry else None)
            print("is on air" if is_on_air else "is_grounded")
            print("depth", depth)

        for i, coords_point_to in enumerate(
                cur_space.available_moves + cur_coords):
            point_to = Point(*coords_point_to)
            space_to = graph.nodes[point_to]["space"]

            if point_to.is_bad(board.get_size()):
                continue
            if not is_available_space(space_to):
                continue
            if not is_on_air and not (space_down.name == "PIPE" or space_down.name == "LADDER") and \
                    (cur_space.available_moves[i] == Direction.down).all():
                continue
            if is_on_air and (cur_space.available_moves[i] != Direction.down).all():
                continue

            if verbose:
                print(f"\t{get_ij(point_to)}")
            graph.add_edge(cur_point, point_to)

        for point_to in graph[cur_point]:
            task_stack.append((point_to, new_depth))

    report = dict()
    for object_name, points in met_objects.items():
        if report.get(object_name) is None:
            report[object_name] = defaultdict(int)
        for p in points:
            report[object_name][closest_distances[p]] += 1
    return graph, met_objects, report


def main():
    board_str = open("./resource/board_example", "r").read()
    board = Board(board_str)
    print(board.get_size())
    graph, sp = create_graph_no_edges(board)

    print(len(graph.nodes()))

    p = Point(42, 33)
    # p = Point(56, 1)
    # p = sp
    print(p)

    graph, met_objects, report = fulfill_graph_edges_from_point(graph, board, p, max_depth=21, verbose=True)
    print(get_ij(p))
    print(report)


if __name__ == "__main__":
    main()
