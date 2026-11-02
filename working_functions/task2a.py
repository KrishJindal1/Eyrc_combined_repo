'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
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

# Team ID:          < 1221 >
# Author List:      <krish Jindal >
# Filename:         task2a.py
# Functions:        detect_lane, compute_control
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import math
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'

LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)
##############################################################
YELLOW_LOWER = np.array([15, 70, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)

WHITE_LOWER = np.array([0, 0, 170], dtype=np.uint8)
WHITE_UPPER = np.array([180, 90, 255], dtype=np.uint8)

ROI_TOP_RATIO = 0.45
ROI_BOTTOM_RATIO = 0.95

NUMBER_OF_BANDS = 12
MIN_PIXELS_PER_BAND = 8
MIN_CURVE_POINTS = 4
#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
###################PID CONTROLLER CONSTANTS#####################
# # Controller gains
# KP = 4.0
# KI = 0.0
# KD = 0.4
# # Gain used to correct the vehicle's heading
# HEADING_GAIN = 1.5

# # Steering limits
# MAX_STEERING_ANGLE = 0.8

# # PID state retained between control steps
# previous_lateral_error = 0.0
# accumulated_lateral_error = 0.0
# Controller gains
KP = 0.002
KI = 0.0
KD = 0.0001
HEADING_GAIN = 0.5
MAX_STEERING_ANGLE = 0.8

# PID state retained between control steps
previous_lateral_error = 0.0
accumulated_lateral_error = 0.0

##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Work out, from a single camera frame, how wrong the vehicle currently is -
    and hand back whatever `compute_control()` needs in order to correct it.

    This is your Task 1C detector, extended. Task 1C answered *where the lane
    centre is* and *which lane the vehicle is in*; a controller cannot use
    either of those directly. It needs to know how far the vehicle is from
    where it should be, and which way it is pointing.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        One BGR frame from the vehicle's camera, of shape (height, width, 3),
        the right way up and in the right colour order.

    Returns:
    ---
    Anything you like - a tuple, a dictionary, your own class. Whatever you
    return here is passed straight to compute_control() and nowhere else, so
    the two functions only have to agree with each other.

    Example call:
    ---
    detection = detect_lane(frame)

    REMEMBER:
    ---
    Both lanes are driven during evaluation, and the arena looks mirrored from
    the other one. Do not hard-code which side the boundary is on.

    Do NOT call cv2.imshow() or cv2.imwrite() from inside this function.

    You are marked on how closely you hold the MIDDLE of your lane, not merely
    on staying the correct side of the dashed line.
'''
    # ADD YOUR CODE HERE
        #################### ADD YOUR CODE HERE ####################
    center_x = -1
    lane = LANE_UNKNOWN

    height, width = frame.shape[:2]
 
    # ---------------------------------------------------------------
    # 1. Isolate yekjhllow and white lane markings.
    # ---------------------------------------------------------------

    yellow_mask, white_mask = _create_lane_masks(
        frame
    )

    # ---------------------------------------------------------------
    # 2. Restrict detection to the road area.
    # ---------------------------------------------------------------

    road_roi = _create_road_roi(
        height,
        width
    )

    yellow_mask = cv2.bitwise_and(
        yellow_mask,
        road_roi
    )

    white_mask = cv2.bitwise_and(
        white_mask,
        road_roi
    )

    top_y = int(
        height * ROI_TOP_RATIO
    )

    bottom_y = int(
        height * ROI_BOTTOM_RATIO
    )

    # ---------------------------------------------------------------
    # 3. Extract points from horizontal bands.
    # ---------------------------------------------------------------

    yellow_points = _get_band_points(
        yellow_mask,
        top_y,
        bottom_y
    )

    white_points = _get_band_points(
        white_mask,
        top_y,
        bottom_y
    )
    # if not hasattr(detect_lane, "debug_count"):
    #     detect_lane.debug_count = 0

    # if detect_lane.debug_count < 10:
    #     print("YELLOW:", yellow_points)
    #     print("WHITE :", white_points)
    #     detect_lane.debug_count += 1
    # ---------------------------------------------------------------
    # 4. Separate the two yellow boundaries.
    #
    # The yellow boundary on the left is to the left of the image
    # centre, while the right boundary is to the right.
    # ---------------------------------------------------------------

    image_center = width / 2.0

    left_yellow_points = [
        point
        for point in yellow_points
        if point[0] <= image_center
    ]

    right_yellow_points = [
        point
        for point in yellow_points
        if point[0] >= image_center
    ]

    # ---------------------------------------------------------------
    # 5. Fit quadratic curves.
    # ---------------------------------------------------------------

    left_yellow_curve = _fit_lane_curve(
        left_yellow_points
    )

    right_yellow_curve = _fit_lane_curve(
        right_yellow_points
    )

    white_curve = _fit_lane_curve(
        white_points
    )



    # ---------------------------------------------------------------
    # 6. Estimate the vehicle's horizontal position.
    # ---------------------------------------------------------------

    vehicle_x = width / 2.0

    # ---------------------------------------------------------------
    # 7. Determine the lane from the white centre line.
    # ---------------------------------------------------------------

    # Use a row where the white line is normally visible.
    lane_reference_y = int(
        height * 0.63
    )

    white_x = _evaluate_curve(
        white_curve,
        lane_reference_y
    )
    # if white_x is not None and abs(white_x - 500) > 60:
    #     print("WHITE POINTS:", white_points)
    #     print("WHITE CURVE:", white_curve)
    #     print("WHITE X:", white_x)

    
    if white_x is None:
        return {
            "center_x": -1,
            "lane": LANE_UNKNOWN,
            "image_width": width
        }
    

    
    if vehicle_x < white_x:
        lane = LANE_LEFT
    else:
        lane = LANE_RIGHT

    # ---------------------------------------------------------------
    # 8. Find the boundaries of the current lane.
    # ---------------------------------------------------------------

    if lane == LANE_LEFT:

        left_yellow_x = _evaluate_curve(
            left_yellow_curve,
            lane_reference_y
        )

        if left_yellow_x is None:
            return {
                "center_x": -1,
                "lane": LANE_UNKNOWN,
                 "image_width": width
            }

        lane_center_x = (
            left_yellow_x + white_x
        ) / 2.0

    elif lane == LANE_RIGHT:

        right_yellow_x = _evaluate_curve(
            right_yellow_curve,
            lane_reference_y
        )

        if right_yellow_x is None:
            return {
                "center_x": -1,
                "lane": LANE_UNKNOWN,
                 "image_width": width
            }

        lane_center_x = (
            white_x + right_yellow_x
        ) / 2.0

    else:

        return {
            "center_x": -1,
            "lane": LANE_UNKNOWN,
            "image_width": width
        }

    # ---------------------------------------------------------------
    # 9. Convert the lane centre to an integer pixel coordinate.
    #
    # No perspective transformation was used, so this is already
    # in the original 640x480 camera coordinate system.
    # ---------------------------------------------------------------

    center_x = int(
        round(lane_center_x)
    )

    
    # ---------------------------------------------------------------
    # 10. Validate the result.
    # ---------------------------------------------------------------

    if not (0 <= center_x < width):
        center_x = max(0, min(width - 1, center_x))
    ############################################################

    return {
    "center_x": center_x,
    "lane": lane,
    "image_width": width,
    "lane_center_y": lane_reference_y,
    "left_yellow_curve": left_yellow_curve,
    "right_yellow_curve": right_yellow_curve,
    "white_curve": white_curve
}



def _create_lane_masks(frame):
    """
    Isolate yellow and white road markings using HSV colour ranges.
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    yellow_mask = cv2.inRange(
        hsv,
        YELLOW_LOWER,
        YELLOW_UPPER
    )

    white_mask = cv2.inRange(
        hsv,
        WHITE_LOWER,
        WHITE_UPPER
    )

    # Remove small isolated noise and close small gaps.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )
    white_mask = _remove_vehicle_from_white_mask(
    white_mask
    )

    return yellow_mask, white_mask


def _create_road_roi(height, width):
    """
    Create a trapezoidal road region to exclude the building,
    sky and vehicle body from lane detection.
    """

    roi = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    top_y = int(height * ROI_TOP_RATIO)
    bottom_y = int(height * ROI_BOTTOM_RATIO)

    polygon = np.array([
        [
            (0, bottom_y),
            (width - 1, bottom_y),
            (width - 1, top_y),
            (0, top_y)
        ]
    ], dtype=np.int32)

    cv2.fillPoly(
        roi,
        polygon,
        255
    )

    return roi


def _get_band_points(mask, top_y, bottom_y):
    """
    Extract one representative x-position from each horizontal band.

    This allows the dashed white line to contribute separate visible
    pieces instead of requiring one continuous marking.
    """

    points = []

    band_height = (
        bottom_y - top_y
    ) // NUMBER_OF_BANDS

    for band_index in range(NUMBER_OF_BANDS):

        y1 = (
            bottom_y
            - (band_index + 1) * band_height
        )

        y2 = (
            bottom_y
            - band_index * band_height
        )

        y1 = max(top_y, y1)
        y2 = min(bottom_y, y2)

        if y1 >= y2:
            continue

        band = mask[y1:y2, :]

        y_pixels, x_pixels = np.nonzero(
            band
        )

        if len(x_pixels) < MIN_PIXELS_PER_BAND:
            continue

        # Median is more resistant to small noisy pixels than mean.
        x_position = float(
            np.median(x_pixels)
        )

        y_position = float(
            np.median(y_pixels + y1)
        )

        points.append(
            (x_position, y_position)
        )

    return points


def _fit_lane_curve(points):
    if len(points) < MIN_CURVE_POINTS:
        return None

    x_values = np.array(
        [point[0] for point in points],
        dtype=np.float64
    )

    y_values = np.array(
        [point[1] for point in points],
        dtype=np.float64
    )

    try:
        # Initial quadratic fit.
        initial_curve = np.polyfit(
            y_values,
            x_values,
            2
        )

        predicted_x = np.polyval(
            initial_curve,
            y_values
        )

        errors = np.abs(
            x_values - predicted_x
        )

        # Remove points that are far from the fitted curve.
        threshold = max(
            25.0,
            2.5 * np.median(errors)
        )

        valid = errors <= threshold

        if np.sum(valid) < MIN_CURVE_POINTS:
            return initial_curve

        return np.polyfit(
            y_values[valid],
            x_values[valid],
            2
        )

    except (
        np.linalg.LinAlgError,
        ValueError,
        TypeError
    ):
        return None


def _evaluate_curve(curve, y):
    """
    Calculate the x-position of a fitted curve at image row y.
    """

    if curve is None:
        return None

    return float(
        np.polyval(
            curve,
            y
        )
    )
def _remove_vehicle_from_white_mask(mask):
    """
    Remove the large white vehicle body from the bottom-centre
    of the white detection mask.
    """

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    height, width = mask.shape

    for label in range(1, number_of_labels):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        touches_bottom = (
            y + component_height >= height
        )

        overlaps_vehicle_region = (
            x < int(width * 0.75)
            and
            x + component_width > int(width * 0.25)
        )

        if (
            touches_bottom
            and overlaps_vehicle_region
            and area > 500
        ):
            mask[labels == label] = 0

    return mask



def compute_control(detection, stop_requested):
#     '''
#     Purpose:
#     ---
#     Turn the errors from detect_lane() into the three numbers the vehicle is
#     driven with, and obey the stop flag.

#     This is where your Task 1B controller and your Task 1A steering geometry
#     meet: work out one steering angle from the error, then split it into the
#     two front-wheel angles the joints actually take.

#     Input Arguments:
#     ---
#     `detection` :   [ any ]
#         Whatever detect_lane() returned for this frame.

#     `stop_requested` :  [ bool ]
#         True while the evaluation is holding the vehicle at a stop point.
#         This is a LEVEL, not a pulse: it stays True for the whole duration of
#         the stop and goes False when the wait is over. While it is True the
#         vehicle must be STATIONARY, so return a speed of 0.0. There is no
#         stopwatch for you to write and no duration to hard-code.

#     Returns:
#     ---
#     `left_angle`, `right_angle` :   [ float ]
#         The two front-wheel steering angles, in RADIANS, written straight to
#         the steering joints. The scene's steering joints are POSITIVE TO THE
#         LEFT. Nothing is clamped for you - the vehicle's own limit is 45
#         degrees, and commanding past it will not steer any harder.

#     `speed` :   [ float ]
#         Target velocity for both drive motors, in rad/s. 0.0 holds the
#         vehicle still.

#     Example call:
#     ---
#     left_angle, right_angle, speed = compute_control(detection, stop_requested)

#     REMEMBER:
#     ---
#     Keep the loop running while stopped - set the SPEED to zero, not your
#     program. If you stop reading frames you will not see the flag drop.

#     Sooner or later a frame comes back with nothing usable in it. Holding the
#     last steering angle, easing off the throttle and stopping dead are all
#     defensible - doing nothing is not.
#     '''

#     # ADD YOUR CODE HERE
    global previous_lateral_error, accumulated_lateral_error

    # NEW: obey the evaluator's stop signal
    if stop_requested:
        previous_lateral_error = 0.0
        accumulated_lateral_error = 0.0
        left_angle, right_angle = ackermann_wheel_angles(0.0)
        return left_angle, right_angle, 0.0

 # NEW: read the values returned by detect_lane()
    center_x = detection["center_x"]
    lane = detection["lane"]
    image_width = detection["image_width"]
 # NEW: handle failed lane detection
    if center_x < 0 or lane == LANE_UNKNOWN:
        left_angle, right_angle = ackermann_wheel_angles(0.0)
        return left_angle, right_angle, 0.0

        # Get the detected lane curves
    white_curve = detection.get("white_curve")
    left_curve = detection.get("left_yellow_curve")
    right_curve = detection.get("right_yellow_curve")
    reference_y = detection["lane_center_y"]

    # Choose the curve that defines the detected lane's direction
    if lane == LANE_LEFT and left_curve is not None:
        lane_curve = left_curve
    elif lane == LANE_RIGHT and right_curve is not None:
        lane_curve = right_curve
    elif white_curve is not None:
        lane_curve = white_curve
    else:
        lane_curve = None

    # Calculate the heading error only if the curves are available
    current_yaw = 0.0

    if white_curve is not None:
        reference_y = detection["lane_center_y"]

        # Calculate the slope of the detected white boundary
        white_slope = np.polyval(
            np.polyder(white_curve), reference_y
        )

        # Convert the image slope into a heading angle
        current_yaw = np.arctan(white_slope)

   

  

    # NEW: calculate lateral error from the image centre
    target_y = image_width / 2.0
    current_y = center_x
    time_step = 0.05  # temporary fixed timestep

    # Positive error means the vehicle is to the right of the target.
    # Positive steering moves it towards smaller y.
    lateral_error = target_y - current_y

    # Accumulate lateral error for the integral term.
    accumulated_lateral_error += lateral_error * time_step

    # Calculate the rate of change of lateral error.
    if time_step > 0:
        lateral_error_rate = (
            lateral_error - previous_lateral_error
        ) / time_step
    else:
        lateral_error_rate = 0.0

    # PID correction based on lateral position.
    steering = (
        KP * lateral_error
        + KI * accumulated_lateral_error
        + KD * lateral_error_rate
    )
    # steering -= HEADING_GAIN * current_yaw
   

    # Save error for the next control step.
    previous_lateral_error = lateral_error

    # Limit the steering command.
    steering = max(
        -MAX_STEERING_ANGLE,
        min(MAX_STEERING_ANGLE, steering)
    )
    # NEW: convert steering into wheel angles and return speed
    left_angle, right_angle = ackermann_wheel_angles(steering)
    speed = 2.0

    return left_angle, right_angle, speed



def ackermann_wheel_angles(delta):
    '''
    Purpose:
    ---
    Convert one desired steering angle into the two real front-wheel angles
    the vehicle's Ackermann geometry implies.

    Input Arguments:
    ---
    `delta` :          [ float ]
        Desired steering angle in radians, as if the vehicle had a single
        centred front wheel. Positive means turning left.

    Returns:
    ---
    `left_wheel_angle`  : [ float ]  angle for the left front wheel, radians
    `right_wheel_angle` : [ float ]  angle for the right front wheel, radians
    '''

    if abs(delta) < 1e-9:
        return 0.0, 0.0

    effective_half_track = (TRACK_WIDTH / 2.0) - WHEEL_OFFSET

    turning_radius = WHEELBASE / math.tan(abs(delta))

    inner_wheel_radius = turning_radius - effective_half_track

    outer_wheel_radius = turning_radius + effective_half_track

    inner_wheel_angle = math.atan(
        WHEELBASE / inner_wheel_radius
    )

    outer_wheel_angle = math.atan(
        WHEELBASE / outer_wheel_radius
    )

    if delta > 0:
        left_angle = inner_wheel_angle
        right_angle = outer_wheel_angle
    else:
        left_angle = -outer_wheel_angle
        right_angle = -inner_wheel_angle




    return left_angle, right_angle

##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
