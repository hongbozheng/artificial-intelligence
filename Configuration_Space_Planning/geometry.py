# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien

def does_alien_touch_wall(alien, walls, granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endy), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    # get the radius/width of the alien
    radius = alien.get_width()
    # check if alien is in circle shape
    if alien.is_circle():
        # get the coordinate of alien's center position
        center = alien.get_centroid()
        for coord in walls:
            # check if the distance from the center of the circle to walls is less than radius + granularity/sqrt(2)
            if abs(pointLineDist(center,(coord[0],coord[1]),(coord[2],coord[3]))) <= radius+granularity/math.sqrt(2):
                return True
        return False
    else:
        # check if the alien (in rectangle shape) touches the walls
        return checkRectAlien(alien, walls, granularity)

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """

    radius = alien.get_width()
    if alien.is_circle():
        center = alien.get_centroid()
        if center[0] <= radius+granularity/math.sqrt(2) or (window[0]-center[0]) <= radius+granularity/math.sqrt(2):
            return False
        if center[1] <= radius+granularity/math.sqrt(2) or (window[1]-center[1]) <= radius+granularity/math.sqrt(2):
            return False
        return True
    else:
        walls = [(0,0,window[0],0),(0,0,0,window[1]),(window[0],window[1],window[0],0),(window[0],window[1],0,window[1])]
        return not checkRectAlien(alien,walls,granularity)

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals

        Return:
            True if a goal is touched, False if not.
    """
    if alien.is_circle():
        center = alien.get_centroid()
        radius = alien.get_width()
        for goal in goals:
            if vectorMagnitude(center,(goal[0],goal[1])) <= radius+goal[2]:
                return True
        return False
    else:
        radius = alien.get_width()
        if alien.get_head_and_tail()[0][1] == alien.get_head_and_tail()[1][1]:
            leftPt = alien.get_head_and_tail()[1]
            rightPt = alien.get_head_and_tail()[0]
            TL = (leftPt[0], leftPt[1] - radius)
            TR = (rightPt[0], rightPt[1] - radius)
            BL = (leftPt[0], leftPt[1] + radius)
            BR = (rightPt[0], rightPt[1] + radius)
            head = leftPt
            tail = rightPt
        else:
            topPt = alien.get_head_and_tail()[0]
            btmPt = alien.get_head_and_tail()[1]
            TL = (topPt[0]-radius,topPt[1])
            TR = (topPt[0]+radius,topPt[1])
            BL = (btmPt[0]-radius,btmPt[1])
            BR = (btmPt[0]+radius,btmPt[1])
            head = topPt
            tail = btmPt

        for goal in goals:
            if abs(pointLineDist((goal[0],goal[1]),TL,TR)) <= goal[2]:
                return True
            if abs(pointLineDist((goal[0],goal[1]),TL,BL)) <= goal[2]:
                return True
            if abs(pointLineDist((goal[0],goal[1]),BR,TR)) <= goal[2]:
                return True
            if abs(pointLineDist((goal[0],goal[1]),BR,BL)) <= goal[2]:
                return True
            if vectorMagnitude(head,(goal[0],goal[1])) <= radius+goal[2]:
                return True
            if vectorMagnitude(tail,(goal[0],goal[1])) <= radius+goal[2]:
                return True

    return False


def dotProduct(pt, A, B):
    """
    Calculate the dot product of line ptA and AB
    :param pt: the point that we are interested
    :param A:  one endpoint of the line segment
    :param B:  the other endpoint of the line segment
    :return:   the result of dot product in float
    """
    vectorA = (pt[0]-A[0],pt[1]-A[1])
    vectorB = (B[0]-A[0],B[1]-A[1])
    return float(vectorA[0]*vectorB[0]+vectorA[1]*vectorB[1])

def vectorMagnitude(A, B):
    """
    Calculate the magnitude of a vector
    :param A: one endpoint of the vector
    :param B: the other endpoint of the vector
    :return:  the magnitude of the vector in float
    """
    vector = (A[0]-B[0],A[1]-B[1])
    return float(math.sqrt(pow(vector[0],2)+pow(vector[1],2)))

def aSinAlpha(pt, A, B):
    """
    Calculate the orthogonal distance from pt to line segment AB
    :param pt: the point that we are interested
    :param A:  one endpoint of the line segment
    :param B:  the other endpoint of the line segment
    :return:   the orthogonal distance from pt to the line segment AB in float
    """
    vectorA = (pt[0]-A[0],pt[1]-A[1])
    vectorB = (B[0]-A[0],B[1]-A[1])
    if vectorMagnitude(A,B) != 0:
        return float((vectorA[0]*vectorB[1]-vectorA[1]*vectorB[0])/vectorMagnitude(A,B))
    else:
        return 0

def pointLineDist(pt, A, B):
    """
    Calculate the the distance from a point to a line segment
    :param pt: the point that we are interested
    :param A:  one endpoint of the line segment
    :param B:  the other endpoint of the line segment
    :return:   the distance between the point and the line segment
    """
    if dotProduct(pt,A,B) <= 0:
        return vectorMagnitude(pt,A)
    if dotProduct(pt,B,A) <= 0:
        return vectorMagnitude(pt,B)
    return aSinAlpha(pt,A,B)

def rectLineDist(TL, TR, BL, BR, A, B):
    """
    Calculate the shortest distance between a rectangle and a line segment
    :param TL: TopLeft coordinate of the rectangle
    :param TR: TopRight coordinate of the rectangle
    :param BL: BottomLeft coordinate of the rectangle
    :param BR: BottomRight coordinate of the rectangle
    :param A: left endpoint of the line segment
    :param B: right endpoint of the line segment
    :return:  the shortest distance between a rectangle and a line segment in float
    """
    dist = []
    dist.append(abs(pointLineDist(TL,A,B)))
    dist.append(abs(pointLineDist(TR,A,B)))
    dist.append(abs(pointLineDist(BL,A,B)))
    dist.append(abs(pointLineDist(BR,A,B)))
    dist.append(abs(pointLineDist(A,TL,TR)))
    dist.append(abs(pointLineDist(A,TL,BL)))
    dist.append(abs(pointLineDist(A,BR,TR)))
    dist.append(abs(pointLineDist(A,BR,BL)))
    dist.append(abs(pointLineDist(B,TL,TR)))
    dist.append(abs(pointLineDist(B,TL,BL)))
    dist.append(abs(pointLineDist(B,BR,TR)))
    dist.append(abs(pointLineDist(B,BR,BL)))
    return min(dist)

def lineLineCross(A, B, C, D):
    """
    Check if two line segment intersect with each other
    :param A: one endpoint of line segment 1
    :param B: the other endpoint of line segment 1
    :param C: one endpoint of line segment 2
    :param D: the other endpoint of line segment 2
    :return:  True if two line segments intersect with each other False Otherwise
    """
    if aSinAlpha(A,C,D) > 0 and aSinAlpha(B,C,D) < 0 or aSinAlpha(A,C,D) < 0 and aSinAlpha(B,C,D) > 0:
        if aSinAlpha(C,A,B) > 0 and aSinAlpha(D,A,B) < 0 or aSinAlpha(C,A,B) < 0 and aSinAlpha(D,A,B) > 0:
            return True
    return False

def lineRectCross(TL, TR, BL, BR, A, B):
    """
    Check if the rectangle intersects with a line segment
    :param TL: TopLeft coordinate of the rectangle
    :param TR: TopRight coordinate of the rectangle
    :param BL: BottomLeft coordinate of the rectangle
    :param BR: BottomRight coordinate of the rectangle
    :param A: left endpoint of the line segment
    :param B: right endpoint of the line segment
    :return:  True if the rectangle intersects with the line segment False Otherwise
    """
    if lineLineCross(TL,TR,A,B):
        return True
    if lineLineCross(TL,BL,A,B):
        return True
    if lineLineCross(BR,TR,A,B):
        return True
    if lineLineCross(BR,BL,A,B):
        return True
    return False

def checkRectAlien(alien, walls, granularity):
    """
    Check if the rectangle alien touches the walls or the windows
    :param alien (Alien): Instance of Alien class that will be navigating our map
    :param walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endy), ...]
    :param granularity (int): The granularity of the map
    :return: True if the alien touches the wall or the windows
    """
    radius = alien.get_width()
    if alien.get_head_and_tail()[0][1] == alien.get_head_and_tail()[1][1]:
        leftPt = alien.get_head_and_tail()[1]
        rightPt = alien.get_head_and_tail()[0]
        TL = (leftPt[0],leftPt[1]-radius)
        TR = (rightPt[0],rightPt[1]-radius)
        BL = (leftPt[0],leftPt[1]+radius)
        BR = (rightPt[0],rightPt[1]+radius)
        head = leftPt
        tail = rightPt
    else:
        topPt = alien.get_head_and_tail()[0]
        btmPt = alien.get_head_and_tail()[1]
        TL = (topPt[0]-radius,topPt[1])
        TR = (topPt[0]+radius,topPt[1])
        BL = (btmPt[0]-radius,btmPt[1])
        BR = (btmPt[0]+radius,btmPt[1])
        head = topPt
        tail = btmPt

    for coord in walls:
        if lineRectCross(TL,TR,BL,BR,(coord[0],coord[1]),(coord[2],coord[3])):
            return True
        if rectLineDist(TL,TR,BL,BR,(coord[0],coord[1]),(coord[2],coord[3])) <= granularity/math.sqrt(2):
            return True
        if abs(pointLineDist(head,(coord[0],coord[1]),(coord[2],coord[3]))) <= radius+granularity/math.sqrt(2):
            return True
        if abs(pointLineDist(tail,(coord[0],coord[1]),(coord[2],coord[3]))) <= radius+granularity/math.sqrt(2):
            return True
    return False

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0)
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner

                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal

                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")