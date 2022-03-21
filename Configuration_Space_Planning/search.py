# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze, ispart1=False):
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None.

    @param maze: Maze instance from maze.py

    @param ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # get the start of the maze
    start = maze.getStart()
    # if start is the goal, return start position
    if maze.isObjective(start[0], start[1],start[2], ispart1):
        return [start]

    # queue to store points
    queue = []
    # set to store visited points
    visited = set()

    # append start into queue and set
    queue.append(start)
    visited.add(start)
    # store the parent node of the children node
    parent_node = {}

    # while queue is not empty
    while queue:
        # pop the element in queue to x
        x = queue.pop(0)

        # get all x's neighbors
        for i in maze.getNeighbors(x[0], x[1], x[2], ispart1):
            # if x's neighbors are not in set visited
            if i not in visited:
                # record its parent node
                parent_node[i] = x
                # check if x is one of the objectives
                if maze.isObjective(i[0], i[1], i[2], ispart1):
                    # return solution path
                    return getSolutionPath(start, i, parent_node)
                # append it into the queue and set
                queue.append(i)
                visited.add(i)

def getSolutionPath(start, end, parent_node):
    """
    Compute the solution path of the maze

    @param start: The start coordinate of the maze
    @param end: The destination coordinate of the maze
    @param parent_node: Dictionary storing the parent node of its corresponding children nodes
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # create list to store solution path
    # append the end position into the list
    path = [end]

    # if the last element stored in path is not the starting position,
    # find the parent node of the last element stored in path
    # until the last element stored in path is the starting position
    while path[-1] != start:
        path.append(parent_node[path[-1]])

    # reverse path since the solution path stored in path is reversed
    # path[0] is the end position, and path[-1] is the starting position
    path.reverse()
    return path