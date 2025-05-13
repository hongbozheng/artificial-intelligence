# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import queue

# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...

class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): manhattan_distance(i, j)
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # store nodes and neighbors using queue
    queue = []
    # store visited positions
    visited = set()
    # append the start position into the queue and set
    queue.append(maze.start)
    visited.add(maze.start)

    # store the parent node of the children node
    parent_node = {}

    while queue:
        # set x = the parent node
        x = queue.pop(0)

        # check if the starting position is the destination
        if x == maze.waypoints[0]:
            return getSolutionPath(maze.start, maze.waypoints[0], parent_node)

        # find the children nodes of the position x
        for i in maze.neighbors(x[0], x[1]):
            # check if the children node is already in the queue
            if i not in visited:
                # append the parent node of x of the children node i into the dictionary
                # children node: parent node
                parent_node[i] = x
                # check if the children node is the destination
                if i == maze.waypoints[0]:
                    return getSolutionPath(maze.start, maze.waypoints[0], parent_node)
                # append the children node into the queue
                queue.append(i)
                # add the children node into the set
                visited.add(i)
    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # store nodes and neighbors using priority queue
    p_queue = queue.PriorityQueue()
    # store visited positions
    visited = set()
    # append the start position into the priority queue and set
    manhattan_distance_coordinate = (manhattan_distance(maze.start, maze.waypoints[0]), maze.start)
    p_queue.put(manhattan_distance_coordinate)
    visited.add(maze.start)

    # store the parent node of the children node
    parent_node = {}

    while p_queue:
        # get the data from the priority queue
        # (manhattan_distance, coordinate)
        data = p_queue.get()
        # set x = the coordinate
        x = data[1]

        if x == maze.waypoints[0]:
            return getSolutionPath(maze.start, maze.waypoints[0], parent_node)

        # find the children nodes of the position x
        for i in maze.neighbors(x[0], x[1]):
            # check if the children node is already in the priority queue
            if i not in visited:
                # append the parent node of x of the children node i into the dictionary
                # children node: parent node
                parent_node[i] = x
                # compute the distance from start to goal via this state
                # distance = distance from start to current state + heuristic estimate of the distance
                children_node_distance_coordinate = (len(getSolutionPath(maze.start, i, parent_node))
                                                     + manhattan_distance(i, maze.waypoints[0]), i)
                # append the children node into the priority queue
                p_queue.put(children_node_distance_coordinate)
                # add the children node into the set
                visited.add(i)
    return []

def astar_multiple(maze):
    """
    Runs suboptimal search algorithm for part 3.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # create priority queue to store states
    p_queue = queue.PriorityQueue()
    # create set to store visited position
    visited = set()
    # store all the waypoints into list
    waypoints = [i for i in maze.waypoints]

    # find the closest_waypoint to the current position (maze.start)
    closest_waypoint = find_closest_waypoint(maze.start, waypoints)
    # compute the f(s) = g(s) + h(s)
    manhattan_heuristic_coordinate = (0 + heuristic(maze.start, closest_waypoint, maze.waypoints),
                                      (maze.start, maze.waypoints))
    # add the state into priority queue
    p_queue.put(manhattan_heuristic_coordinate)
    # add the state into visited
    visited.add((maze.start, maze.waypoints))

    # dictionary used to store parent_nodes
    parent_node = {}
    # tuple used to store the final waypoint
    final_waypoint = ()
    # list used to store the final solution path
    solution_path = []

    while p_queue:
        # get the state from the priority queue
        # state information = (weight, ((current position), (remain waypoints)))
        data = p_queue.get()

        # check if the remain waypoints is empty
        # if it's empty the solution path is found since all the waypoints are reached
        if not data[1][1]:
            path = getSolutionPath((maze.start, maze.waypoints), final_waypoint, parent_node)
            for i in path:
                solution_path.append(i[0])
            return solution_path

        # if remain waypoints is not empty
        else:
            # get the x, y value of the current position
            for i in maze.neighbors(data[1][0][0], data[1][0][1]):

                # check if the current position (children node) is in remain waypoints of its parent node
                # if it's in, a waypoint is reached. remove the waypoint from the remain waypoints
                if i in data[1][1]:
                    remain_waypoints = list(data[1][1])
                    remain_waypoints.remove(i)

                    # check if the children node is in visited
                    if (i, tuple(remain_waypoints)) not in visited:
                        # add it into dictionary
                        parent_node[(i, tuple(remain_waypoints))] = data[1]

                        # check if remain_waypoints is empty
                        # if it's empty means the last waypoint just reached. Set the final waypoint to this position
                        if not remain_waypoints:
                            final_waypoint = (i, tuple(remain_waypoints))

                        # find closest_waypoint
                        closest_waypoint = find_closest_waypoint(i, remain_waypoints)
                        # calculate children state information
                        children = (len(getSolutionPath((maze.start, maze.waypoints),
                                                                 (i, tuple(remain_waypoints)), parent_node))
                                        + heuristic(i, closest_waypoint, remain_waypoints),
                                        (i, tuple(remain_waypoints)))

                        # add children node into the priority queue
                        p_queue.put(children)
                        # add children node into visited
                        visited.add((i, tuple(remain_waypoints)))

                # check if the current position (children node) is in remain waypoints of its parent node
                # if it's NOT in, NO waypoint is reached. Keep remain waypoints as its parent node
                else:
                    remain_waypoints = data[1][1]

                    if (i, remain_waypoints) not in visited:
                        parent_node[(i, remain_waypoints)] = data[1]

                        closest_waypoint = find_closest_waypoint(i, remain_waypoints)
                        children = (len(getSolutionPath((maze.start, maze.waypoints), (i, remain_waypoints),
                                                             parent_node))
                                    + heuristic(i, closest_waypoint, remain_waypoints), (i, remain_waypoints))

                        p_queue.put(children)
                        visited.add((i, remain_waypoints))

    return []

########################################################################################################################
###################################### IMPLEMENT USING WEIRD ASS SEARCH ALGORITHM ######################################
########################################################################################################################
############ NOT USING HEURISTIC FUNCTION (DISTANCE(current_state, closest_waypoint) + MST(remain_waypoints)) ##########
########################################################################################################################
############################### ONLY USING g FUNCTION (DISTANCE(maze.start, current_state)) ############################
########################################################################################################################
######################################## NO FUCKING IDEA WHY THE HELL IT WORKS #########################################
########################################################################################################################

# def fast(maze):
#     waypoints = [i for i in maze.waypoints]
#     start_point = maze.start
#     final_result = [maze.start]
#
#     while waypoints:
#         path = getFastSearchPath(maze, start_point, waypoints)
#         start_point = path[-1]
#         if start_point in waypoints:
#             waypoints.remove(path[-1])
#         final_result += path
#     return final_result
#
# def getFastSearchPath(maze, start, waypoints):
#     remain_waypoints = [i for i in waypoints]
#     p_queue = queue.PriorityQueue()
#     parent_node = {}
#     distance = {}
#     visited = []
#     distance[start] = 0
#     p_queue.put((distance[start], start))
#     visited.append(start)
#     current_state = p_queue.get()
#     end_point = remain_waypoints[0]
#
#     while current_state[1] not in waypoints:
#         neighbor = maze.neighbors(current_state[1][0], current_state[1][1])
#         for i in neighbor:
#             if i in distance:
#                 score = distance[current_state[1]] + 1
#                 if score < distance[i]:
#                     parent_node[i] = current_state[1]
#                     distance[i] = score
#
#                     if i not in visited:
#                         p_queue.put((distance[i], i))
#             else:
#                 parent_node[i] = current_state[1]
#                 distance[i] = distance[current_state[1]] + 1
#                 if i not in visited:
#                     p_queue.put((distance[i],i))
#                     visited.append(i)
#
#         current_state = p_queue.get()
#
#         if current_state[1] in remain_waypoints:
#             end_point = current_state[1]
#             remain_waypoints.remove(current_state[1])
#
#     solution_path = [end_point]
#     while parent_node[solution_path[-1]] != start:
#         solution_path.append(parent_node[solution_path[-1]])
#     solution_path.reverse()
#     return solution_path

def fast(maze):
    waypoints = [i for i in maze.waypoints]
    start_position = maze.start
    final_path = [maze.start]

    while waypoints:
        remain_waypoints = [i for i in waypoints]
        p_queue = queue.PriorityQueue()
        parent_node = {}
        distance = {}
        visited = []
        distance[start_position] = 0
        p_queue.put((distance[start_position], start_position))
        visited.append(start_position)
        current_state = p_queue.get()
        end_point = remain_waypoints[0]

        while current_state[1] not in waypoints:
            neighbor = maze.neighbors(current_state[1][0], current_state[1][1])
            for i in neighbor:
                if i in distance:
                    score = distance[current_state[1]] + 1
                    if score < distance[i]:
                        parent_node[i] = current_state[1]
                        distance[i] = score

                        if i not in visited:
                            p_queue.put((distance[i], i))
                else:
                    parent_node[i] = current_state[1]
                    distance[i] = distance[current_state[1]] + 1
                    if i not in visited:
                        p_queue.put((distance[i], i))
                        visited.append(i)

            current_state = p_queue.get()

            if current_state[1] in remain_waypoints:
                end_point = current_state[1]
                remain_waypoints.remove(current_state[1])

        solution_path = [end_point]
        while parent_node[solution_path[-1]] != start_position:
            solution_path.append(parent_node[solution_path[-1]])
        solution_path.reverse()
        partial_path = solution_path

        start_position = partial_path[-1]
        if start_position in waypoints:
            waypoints.remove(partial_path[-1])
        final_path += partial_path
    return final_path

########################################################################################################################
############################################ IMPLEMENT USING GREEDY ALGORITHM ##########################################
########################################################################################################################
####################################### WRONG PATH LENGTH WHILE TESTING MEDIUM MAZE ####################################
########################################################################################################################
# def fast(maze):
#
#     """
#     Runs Fast for part 4 of the assignment in the case where there are
#     multiple objectives.
#
#     @param maze: The maze to execute the search on.
#
#     @return path: a list of tuples containing the coordinates of each state in the computed path
#     """
#
#     # store nodes and neighbors using priority queue
#     p_queue = queue.PriorityQueue()
#     # store visited positions
#     visited = set()
#     # store remain waypoint(s)
#     remain_waypoints = list(maze.waypoints)
#     # set start to be the starting position
#     start = maze.start
#     # get the first waypoint from the priority queue
#     waypoint = find_closest_waypoint(start, remain_waypoints)
#     # append the start position into the priority queue and set
#     manhattan_distance_coordinate = (manhattan_distance(start, waypoint), start)
#     p_queue.put(manhattan_distance_coordinate)
#     visited.add(start)
#
#     # store the parent node of the children node
#     parent_node = {}
#     # store the solution path using list
#     solution_path = []
#
#     # loop through all the waypoints
#     for i in range(len(maze.waypoints)):
#
#         while not p_queue.empty():
#             # get the data from the priority queue
#             # (manhattan_distance, coordinate)
#             data = p_queue.get()
#             # set x = the coordinate
#             x = data[1]
#
#             # find the children nodes of the position x
#             for j in maze.neighbors(x[0], x[1]):
#                 # check if the children node is already in the priority queue
#                 if j not in visited:
#                     # append the parent node of x of the children node i into the dictionary
#                     # children node: parent node
#                     parent_node[j] = x
#                     # compute the distance from start to goal via this state
#                     # distance = distance from start to current state + heuristic estimate of the distance
#                     children_node_distance_coordinate = (len(getSolutionPath(start, j, parent_node))
#                                                          + manhattan_distance(j, waypoint), j)
#                     # append the children node into the priority queue
#                     p_queue.put(children_node_distance_coordinate)
#                     # add the children node into the set
#                     visited.add(j)
#
#                     # if the children node is the waypoint
#                     if j == waypoint:
#                         # store the solution path in list as temp variable
#                         path = getSolutionPath(start, waypoint, parent_node)
#                         # add the solution path into list
#                         for j in path:
#                             solution_path.append(j)
#                         # delete the last element in solution path
#                         # since it's used as the starting position of next route
#                         del solution_path[-1]
#                         # reinitialize priority queue, list, and dictionary
#                         p_queue = queue.PriorityQueue()
#                         visited = set()
#                         parent_node = {}
#                         break
#
#         # if the waypoint is not the last waypoint
#         if i < len(maze.waypoints) - 1:
#             # remove the visited waypoint from the list
#             remain_waypoints.remove(waypoint)
#             # set start to be the waypoint
#             start = waypoint
#             # set waypoint to the next closest waypoint
#             waypoint = find_closest_waypoint(start, remain_waypoints)
#             # append the start position into the priority queue and set
#             manhattan_distance_coordinate = (manhattan_distance(start, waypoint), start)
#             p_queue.put(manhattan_distance_coordinate)
#             visited.add(start)
#         # if the waypoint is the last waypoint
#         else:
#             # add the waypoint to the list
#             solution_path.append(waypoint)
#             break
#
#     return solution_path

########################################################################################################################
############################################## Astar_Multiple Version 1.0 ##############################################
########################################################################################################################
################################################ DISCARDED (DO NOT USE) ################################################
########################################################################################################################
################################# GIVE THE WRONG PATH LENGTH WHILE TESTING MEDIUM MAZE #################################
########################################################################################################################

# def astar_multiple(maze):
#     """
#         Runs suboptimal search algorithm for part 3.
#
#         @param maze: The maze to execute the search on.
#
#         @return path: a list of tuples containing the coordinates of each state in the computed path
#         """
#
#     p_queue = queue.PriorityQueue()
#     visited = set()
#     waypoints = [i for i in maze.waypoints]
#
#     closest_waypoint = find_closest_waypoint(maze.start, waypoints)
#
#     manhattan_heuristic_coordinate = (0 + heuristic(maze.start, closest_waypoint, maze.waypoints),
#                                       (maze.start, maze.waypoints))
#     p_queue.put(manhattan_heuristic_coordinate)
#     visited.add((maze.start, maze.waypoints))
#
#     parent_node = {}
#     final_waypoint = ()
#     solution_path = []
#
#     while p_queue:
#         data = p_queue.get()
#
#         if not data[1][1]:
#             path = getAstarSolutionPath((maze.start, maze.waypoints), final_waypoint, parent_node)
#             for i in path:
#                 solution_path.append(i[0])
#             return solution_path
#
#         else:
#             for i in maze.neighbors(data[1][0][0], data[1][0][1]):
#
#                 if i in data[1][1]:
#                     remain_waypoints = list(data[1][1])
#                     remain_waypoints.remove(i)
#
#                     if (i, tuple(remain_waypoints)) not in visited:
#                         parent_node[(i, tuple(remain_waypoints))] = data[1]
#
#                         if not remain_waypoints:
#                             final_waypoint = (i, tuple(remain_waypoints))
#                             children = (len(getAstarSolutionPath((maze.start, maze.waypoints),
#                                                                  (i, tuple(remain_waypoints)),
#                                                                  parent_node)) + 0, (i, tuple(remain_waypoints)))
#                         else:
#                             closest_waypoint = find_closest_waypoint(maze.start, remain_waypoints)
#                             children = (len(getAstarSolutionPath((maze.start, maze.waypoints),
#                                                                  (i, tuple(remain_waypoints)), parent_node))
#                                         + heuristic(i, closest_waypoint, remain_waypoints),
#                                         (i, tuple(remain_waypoints)))
#
#                         p_queue.put(children)
#                         visited.add((i, tuple(remain_waypoints)))
#
#                 else:
#                     remain_waypoints = data[1][1]
#
#                     if (i, remain_waypoints) not in visited:
#                         parent_node[(i, remain_waypoints)] = data[1]
#
#                         closest_waypoint = find_closest_waypoint(maze.start, remain_waypoints)
#                         children = (len(getAstarSolutionPath((maze.start, maze.waypoints), (i, remain_waypoints),
#                                                              parent_node))
#                                     + heuristic(i, closest_waypoint, remain_waypoints), (i, remain_waypoints))
#
#                         p_queue.put(children)
#                         visited.add((i, remain_waypoints))
#
#     return []

########################################################################################################################
############################################## Astar_Multiple Version 1.0 ##############################################
########################################################################################################################
################################################ DISCARDED (DO NOT USE) ################################################
########################################################################################################################
################################################ SUPER LONG RUNNING TIME ###############################################
########################################################################################################################

# def fast(maze):
#     """
#     Runs suboptimal search algorithm for part 4.
#
#     @param maze: The maze to execute the search on.
#
#     @return path: a list of tuples containing the coordinates of each state in the computed path
#     """
#
#     p_queue = queue.PriorityQueue()
#     visited = set()
#     waypoints = [i for i in maze.waypoints]
#
#     closest_waypoint = find_closest_waypoint(maze.start, waypoints)
#
#     manhattan_heuristic_coordinate = (0 + heuristic(maze.start, closest_waypoint, maze.waypoints),
#                                       (maze.start, maze.waypoints))
#     p_queue.put(manhattan_heuristic_coordinate)
#     visited.add((maze.start, maze.waypoints))
#
#     parent_node = {}
#     final_waypoint = ()
#     solution_path = []
#
#     while p_queue:
#         data = p_queue.get()
#
#         if not data[1][1]:
#             path = getAstarSolutionPath((maze.start, maze.waypoints), final_waypoint, parent_node)
#             for i in path:
#                 solution_path.append(i[0])
#             return solution_path
#
#         else:
#             for i in maze.neighbors(data[1][0][0], data[1][0][1]):
#
#                 if i in data[1][1]:
#                     remain_waypoints = list(data[1][1])
#                     remain_waypoints.remove(i)
#
#                     if (i, tuple(remain_waypoints)) not in visited:
#                         parent_node[(i, tuple(remain_waypoints))] = data[1]
#
#                         if not remain_waypoints:
#                             final_waypoint = (i, tuple(remain_waypoints))
#                             children = (len(getAstarSolutionPath((maze.start, maze.waypoints),
#                                                                  (i, tuple(remain_waypoints)),
#                                                      parent_node)) + 0, (i, tuple(remain_waypoints)))
#                         else:
#                             closest_waypoint = find_closest_waypoint(maze.start, remain_waypoints)
#                             children = (len(getAstarSolutionPath((maze.start, maze.waypoints),
#                                                                  (i, tuple(remain_waypoints)), parent_node))
#                                             + heuristic(i, closest_waypoint, remain_waypoints),
#                                         (i, tuple(remain_waypoints)))
#
#                         p_queue.put(children)
#                         visited.add((i, tuple(remain_waypoints)))
#
#                 else:
#                     remain_waypoints = data[1][1]
#
#                     if (i, remain_waypoints) not in visited:
#                         parent_node[(i, remain_waypoints)] = data[1]
#
#                         closest_waypoint = find_closest_waypoint(maze.start, remain_waypoints)
#                         children = (len(getAstarSolutionPath((maze.start, maze.waypoints), (i, remain_waypoints),
#                                                              parent_node))
#                                         + heuristic(i, closest_waypoint, remain_waypoints), (i, remain_waypoints))
#
#                         p_queue.put(children)
#                         visited.add((i, remain_waypoints))
#
#     return []

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

def manhattan_distance(start, end):
    """
    Compute the manhattan distance between two coordinates

    @param start: The coordinate of the starting position
    @param end: The coordinate of the destination
    @return distance: The manhattan distance between two coordinates
    """

    distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
    return distance

def heuristic(current_position, closest_waypoint, remain_waypoints):
    """
    Compute the heuristic function of astar_multiple search
    Heuristic Function (current_position, remain_waypoints) = manhattan_distance(current_position, remain_waypoints)
                                                              + MST(remain_waypoints)
    :param current_position: position of the current state
    :param closest_waypoint: the closest waypoint to the position of the current state
    :param remain_waypoints: remaining waypoints
    :return: integer value of heuristic function
    """

    # No closest waypoint since remain_waypoints is an empty tuple
    # already reach the last waypoint aka already found the solution path
    if closest_waypoint == (-1,-1):
        return 0

    # Compute the integer result of the heuristic function
    else:
        return manhattan_distance(current_position, closest_waypoint) + MST(remain_waypoints).compute_mst_weight()

def find_closest_waypoint(current_position, waypoints):
    """
    Find the closest waypoint to the position of the current state
    :param current_position: position of the current state
    :param waypoints: the waypoints which are used to measured the manhattan distance
                      to the position of the current state
    :return: a tuple of the closest waypoint's coordinate
    """

    # find the coordinate of the closest waypoint only if there are remaining waypoints
    if waypoints:
        waypoints_distances = { i: manhattan_distance(current_position, i) for i in waypoints }
        closest_waypoint = min(waypoints_distances, key=waypoints_distances.get)
        return closest_waypoint

    # return an invalid coordinate as the closest waypoint since the remaining waypoints is empty
    else:
        return (-1,-1)