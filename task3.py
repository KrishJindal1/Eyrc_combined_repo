

# Add global variables and constants here with 
''' ################ like this ###############
seperate each non connected function like this and constants too so easy to check.
'''

##################Import Required Libraries####################
import sys
import time
import math
import numpy as np
import cv2
import os

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

###############Coppelas Simulation Constants####################
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'


STEER_LEFT = '/Niti_Vahan/steeringLeft'
STEER_RIGHT = '/Niti_Vahan/steeringRight'

MOTOR_LEFT = '/Niti_Vahan/steeringLeft/Cylinder/motorLeft'
MOTOR_RIGHT = '/Niti_Vahan/steeringRight/Cylinder/motorRight'





# ========================= PERSON 1 -> krish  =========================

# Sign detection, obstacle detection, lane detection, and visual marker detection.


# ========================= PERSON 2 -> Parminder =========================
# Path tracking/control, sign and obstacle behaviour, parking manoeuvre,
# simulation connection, driving, and main().

# Add Aditional functions required accordingly as many helper functions allowed 





####################Coppelas Simulation Connection####################
def connect_and_locate():
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_3.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}

    for name, path in (
        ('visionSensor', VISION_SENSOR),
        ('steerLeft', STEER_LEFT),
        ('steerRight', STEER_RIGHT),
        ('motorLeft', MOTOR_LEFT),
        ('motorRight', MOTOR_RIGHT),
    ):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Check the Task_3_Easy scene hierarchy.\n'
                f'  Simulator said: {err}')

    

    return sim, handles

