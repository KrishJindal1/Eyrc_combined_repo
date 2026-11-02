'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1A of Niti Vahan (NV) Theme.
*
*  Filename:         ackermann_steering.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         ackermann_steering.py
# Functions:        ackermann_wheel_angles
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import math
import numpy as np
##############################################################


#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################


def ackermann_wheel_angles(delta):
    '''
    Purpose:
    ---
    Convert a single virtual steering angle into the two real front-wheel
    angles, per the Ackermann geometry.

    Input Arguments:
    ---
    `delta` :   [ float ]
        Steering angle of the virtual centred front wheel, in radians.

    Returns:
    ---
    `left_angle`  : [ float ]
    `right_angle` : [ float ]
        The two real front-wheel steering angles, in radians, using the
        same sign convention as delta.

    REMEMBER:
    ---
    WHEEL_OFFSET changes the effective half-track width inside each wheel's triangle.
    '''
   # If the vehicle is travelling straight, both wheels remain straight.
    # A very small tolerance is used to avoid floating-point precision issues.
    if abs(delta) < 1e-9:
        return 0.0, 0.0

    # Effective half-track distance between the vehicle centreline
    # and the steering/kingpin axis of a front wheel.
    #
    # WHEEL_OFFSET accounts for the distance between the kingpin axis
    # and the wheel centre.
    effective_half_track = (TRACK_WIDTH / 2.0) - WHEEL_OFFSET

    # Calculate the turning radius of the virtual centred front wheel.
    #
    # From the Ackermann geometry:
    
    #       tan(|delta|) = WHEELBASE / turning_radius
    
    # Therefore:
    
    #       turning_radius = WHEELBASE / tan(|delta|)
    
    # abs(delta) is used because the radius is always treated as positive.
    turning_radius = WHEELBASE / math.tan(abs(delta))

    # The inner wheel follows a smaller turning radius.
    #
    # Inner radius = turning radius - effective half-track
    inner_wheel_radius = turning_radius - effective_half_track

    # The outer wheel follows a larger turning radius.
    #
    # Outer radius = turning radius + effective half-track
    outer_wheel_radius = turning_radius + effective_half_track

    # Calculate the steering angle of the inner wheel.
    #
    # From the right triangle:
    #
    #       tan(inner_angle) = WHEELBASE / inner_wheel_radius
    #
    # Hence:
    #
    #       inner_angle = atan(WHEELBASE / inner_wheel_radius)
    inner_wheel_angle = math.atan(
        WHEELBASE / inner_wheel_radius
    )

    # Calculate the steering angle of the outer wheel.
    #
    #       tan(outer_angle) = WHEELBASE / outer_wheel_radius
    #
    # Hence:
    #
    #       outer_angle = atan(WHEELBASE / outer_wheel_radius)
    outer_wheel_angle = math.atan(
        WHEELBASE / outer_wheel_radius
    )

    # Assign the inner and outer angles to the physical left/right wheels.
    #
    # Positive delta  -> left turn  -> left wheel is inner
    # Negative delta  -> right turn -> right wheel is inner
    #
    # The sign of the returned angles is kept the same as delta.
    if delta > 0:
        left_angle = inner_wheel_angle
        right_angle = outer_wheel_angle
    else:
        left_angle = -outer_wheel_angle
        right_angle = -inner_wheel_angle



    return left_angle, right_angle


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

if __name__ == "__main__":

    test_angles = np.arange(-0.35, 0.35, 0.05)

    for d in test_angles:
        left, right = ackermann_wheel_angles(d)
        print(f"delta={d}  ->  left={left}, right={right}")
