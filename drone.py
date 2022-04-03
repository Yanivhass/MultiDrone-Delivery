from dataclasses import dataclass
import numpy as np

@dataclass
class Point:
    x: float = 0
    y: float = 0


def turn_to_next_point(curr_point: Point, next_point: Point, standoff_dist: float = 1) -> Point:
    # called when drone 1 reached end of straight line segment
    # make sure all tello speeds are 0
    # calculate new x,y for moving standoff_dist in direction of next_point
    # in parallel:
    #   move drone 1 to new x,y - change yex t degs, move forward standoff_dist
    #   move drone 2 to last point(drone 1 location at the beginning)
    theta = np.arctan((next_point.x - curr_point.x) / (next_point.y - curr_point.y))  # turn angle
    new_point = Point()
    new_point.x = curr_point.x + np.cos(theta) * standoff_dist
    new_point.y = curr_point.y + np.sim(theta) * standoff_dist
    return new_point


class Drone:
    def __init__(self, location: Point, heading=0, standoff_dist=1):
        self.location = location
        self.heading = heading
        self.standoff_dist = standoff_dist
        self.v = Point()

    def turn(self, next_point: Point):
        self.location = turn_to_next_point(self.location, next_point, self.standoff_dist)

