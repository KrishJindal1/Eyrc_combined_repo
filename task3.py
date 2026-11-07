


# Team ID:          < 1221 >
# Author List:      <Krish Jindal , Parmindar Singh >
# Filename:         task3.py
# Functions:        detect_lane, compute_control, _band_points, _ransac_quad, ackermann_wheel_angles
# Global variables: WHEELBASE, TRACK_WIDTH, MAX_STEER, DEFAULT_SPEED, CAMERA_CENTER, REF_ROW,
#                   LOOKAHEAD_ROW, Y_HORIZON, LANE_WIDTH, LANE_WIDTH_FACTOR, Y_TOP, Y_BOT,
#                   N_BANDS, MIN_PIX, WHITE_LO, WHITE_HI, YELLOW_LO, YELLOW_HI, RNG,
#                   METRES_PER_PIXEL, K_STANLEY, K_SOFT, K_HEADING, LAST_STATE


####################### IMPORT MODULES #######################
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
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

import math

WHEELBASE = 0.120        
TRACK_WIDTH = 0.110      
MAX_STEER = math.radians(45.0)  
DEFAULT_SPEED = 1.3      

CAMERA_CENTER = 320.0    # Principal image center x-coordinate for 640x480 resolution (pixels)
REF_ROW = 440.0          # Reference image row near vehicle front axle for lane evaluation (pixels)
LOOKAHEAD_ROW = 320.0    # Lookahead row ahead of vehicle used to compute heading angle psi (pixels)
Y_HORIZON = 240.0        # True camera horizon / vanishing row for 640x480 horizontal forward view
LANE_WIDTH = 530.0       # Measured distance between dashed center line and yellow line at row REF_ROW (single lane)
LANE_WIDTH_FACTOR = LANE_WIDTH / (REF_ROW - Y_HORIZON)  # Linear expansion rate (2.65 px width per vertical row)

Y_TOP = 260              # Top boundary row of the region of interest (strictly below horizon to exclude buildings)
Y_BOT = 450              # Bottom boundary row of the region of interest
N_BANDS = 20             # Number of horizontal strips to divide the ROI for centroid extraction
MIN_PIX = 8              # Minimum active pixels required in a band to accept a centroid

WHITE_LO = np.array([0, 0, 190], dtype=np.uint8)    
WHITE_HI = np.array([180, 50, 255], dtype=np.uint8)  
YELLOW_LO = np.array([20, 100, 100], dtype=np.uint8) 
YELLOW_HI = np.array([35, 255, 255], dtype=np.uint8) 

RNG = np.random.default_rng(0)  # Seeded random number generator for deterministic RANSAC sampling

# Scale factor converting image pixel offsets at REF_ROW to lateral distance in metres
# At REF_ROW (440 px), lane width is ~530 px corresponding to real road lane width ~0.2 m
METRES_PER_PIXEL = 0.20 / 530.0

K_STANLEY = 2.5          # Cross-track error gain for the Stanley steering controller
K_SOFT = 0.2             # Stanley softening term (prevents jitter and numerical instability at low speeds)
K_HEADING = 1.0          # Weight applied to heading error (psi) in steering computation


LAST_STATE = {           
    "steering": 0.0,
    "last_valid_cx": CAMERA_CENTER,
    "lane": None,
    "frame_count": 0
}


def _band_points(mask):
    """Extract centroid (y, x) for horizontal bands across the region of interest."""
    edges = np.linspace(Y_TOP, Y_BOT, N_BANDS + 1).astype(int)
    ys_out, xs_out = [], []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        ys, xs = np.nonzero(mask[y0:y1])
        if len(xs) >= MIN_PIX:
            xs_out.append(xs.mean())
            ys_out.append((y0 + y1) / 2.0)

    # print("ys_out:", ys_out)
    # print("xs_out:", xs_out)
    return np.array(ys_out), np.array(xs_out)


def _ransac_quad(ys, xs, iters=50, tol=8.0, min_in=4):
    """Fit quadratic curve x(y) = a*y^2 + b*y + c using RANSAC to reject outliers."""
    n = len(ys)
    if n < min_in:
        return None
    best = None
    for _ in range(iters):
        idx = RNG.choice(n, 3, replace=False)
        p = np.polyfit(ys[idx], xs[idx], 2)
        inl = np.abs(np.polyval(p, ys) - xs) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_in:
        return None
    fit = np.polyfit(ys[best], xs[best], 2)
    # print("ransac_quad polyfit [a, b, c]:", fit)
    return fit


def ackermann_wheel_angles(delta):
    """
    Split single-track target angle delta into left and right front-wheel angles.
    Positive angle turns the vehicle left.
    """
    if abs(delta) < 1e-6:
        return 0.0, 0.0

    tan_delta = math.tan(delta)
    denom_left = 2.0 * WHEELBASE - TRACK_WIDTH * tan_delta
    denom_right = 2.0 * WHEELBASE + TRACK_WIDTH * tan_delta

    if abs(denom_left) < 1e-6 or abs(denom_right) < 1e-6:
        return delta, delta

    tan_left = (2.0 * WHEELBASE * tan_delta) / denom_left
    tan_right = (2.0 * WHEELBASE * tan_delta) / denom_right

    return math.atan(tan_left), math.atan(tan_right)


def detect_lane(frame):
    """
    Detect lane boundaries, compute lane center at REF_ROW, and calculate 
    heading angle psi geometrically from lookahead lane center.
    """
    global LAST_STATE
    LAST_STATE["frame_count"] += 1
    print(f"\n--- Frame {LAST_STATE['frame_count']} ---")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    white[yellow > 0] = 0

    # 1. Detect dashed center line first
    pw = _ransac_quad(*_band_points(white))
    xw = np.polyval(pw, REF_ROW) if pw is not None else None

    # 2. Lock assigned lane on initial frames so the car never flips lanes mid-run
    if LAST_STATE["lane"] is None:
        if xw is not None:
            LAST_STATE["lane"] = "left" if xw > CAMERA_CENTER else "right"
        else:
            left_y = np.count_nonzero(yellow[:, :int(CAMERA_CENTER)])
            right_y = np.count_nonzero(yellow[:, int(CAMERA_CENTER):])
            LAST_STATE["lane"] = "left" if left_y >= right_y else "right"

    lane = LAST_STATE["lane"]
    print(lane)

    # 3. Filter yellow markings to ONLY track the outer boundary of the assigned lane:
    # - Left Lane: yellow outer boundary is strictly on the left (x < CAMERA_CENTER)
    # - Right Lane: yellow outer boundary is strictly on the right (x > CAMERA_CENTER)
    yellow_lane = yellow.copy()
    if lane == "left":
        yellow_lane[:, int(CAMERA_CENTER):] = 0
    else:
        yellow_lane[:, :int(CAMERA_CENTER)] = 0

    py = _ransac_quad(*_band_points(yellow_lane))
    xy = np.polyval(py, REF_ROW) if py is not None else None

    if pw is None and py is None:
        return {
            "valid": False,
            "center_x": LAST_STATE["last_valid_cx"],
            "psi": 0.0,
            "lane": lane
        }

    # 4. Compute lane center x using calibrated half-lane width at REF_ROW
    half_lane = 0.5 * LANE_WIDTH_FACTOR * (REF_ROW - Y_HORIZON)

    if xw is not None and xy is not None:
        cx = (xw + xy) / 2.0
    elif xw is not None:
        # In Left Lane, yellow boundary is on left -> center is left of white (xw - half_lane)
        # In Right Lane, yellow boundary is on right -> center is right of white (xw + half_lane)
        cx = xw - half_lane if lane == "left" else xw + half_lane
    else:
        # In Left Lane, yellow boundary is on left -> center is right of yellow (xy + half_lane)
        # In Right Lane, yellow boundary is on right -> center is left of yellow (xy - half_lane)
        cx = xy + half_lane if lane == "left" else xy - half_lane

    cx = float(np.clip(cx, 0.0, frame.shape[1] - 1.0))
    LAST_STATE["last_valid_cx"] = cx       
    print("center x:", cx) 

    # 5. Compute heading angle psi geometrically using lookahead row (no polynomial slope derivatives)
    xw_ahead = np.polyval(pw, LOOKAHEAD_ROW) if pw is not None else None
    xy_ahead = np.polyval(py, LOOKAHEAD_ROW) if py is not None else None
    half_lane_ahead = 0.5 * LANE_WIDTH_FACTOR * (LOOKAHEAD_ROW - Y_HORIZON)

    if xw_ahead is not None and xy_ahead is not None:
        cx_ahead = (xw_ahead + xy_ahead) / 2.0
    elif xw_ahead is not None:
        cx_ahead = xw_ahead - half_lane_ahead if lane == "left" else xw_ahead + half_lane_ahead
    elif xy_ahead is not None:
        cx_ahead = xy_ahead + half_lane_ahead if lane == "left" else xy_ahead - half_lane_ahead
    else:
        cx_ahead = cx

    # Perspective compensation:
    # On a straight road, any lane offset from CAMERA_CENTER converges towards CAMERA_CENTER at Y_HORIZON.
    # The expected position of a straight path at LOOKAHEAD_ROW is:
    scale_ahead = (LOOKAHEAD_ROW - Y_HORIZON) / (REF_ROW - Y_HORIZON)
    cx_straight_ahead = CAMERA_CENTER + (cx - CAMERA_CENTER) * scale_ahead

    # Road curvature offset: only true bends deviate from the straight perspective line
    delta_x_curve = cx_ahead - cx_straight_ahead

    # Heading angle psi: orientation of the true road curve relative to vehicle heading
    # In CoppeliaSim, positive steering turns LEFT; negative turns RIGHT.
    psi = -math.atan2(delta_x_curve, REF_ROW - LOOKAHEAD_ROW)
    print("psi (rad):", round(psi, 4))

    return {
        "valid": True,
        "center_x": cx,
        "psi": psi,
        "lane": lane
    }


def compute_control(detection, stop_requested):
    """
    Execute Stanley tracking on visual errors and command front wheel angles & motor speeds.
    """
    global LAST_STATE

    if stop_requested:
        # Full stop: keep steering steady and kill motor speed
        left_angle, right_angle = ackermann_wheel_angles(LAST_STATE["steering"])
        return left_angle, right_angle, 0.0

    if detection is None or not detection.get("valid", False):
        # Graceful fallback: maintain previous steering angle and ease speed
        steer = LAST_STATE["steering"]
        left_angle, right_angle = ackermann_wheel_angles(steer)
        return left_angle, right_angle, DEFAULT_SPEED * 0.8

    cx = detection["center_x"]
    psi = detection.get("psi", 0.0)

    # 1. Lateral Cross-Track Error (e):
    # Positive steering in scene is to the LEFT.
    # If cx < CAMERA_CENTER, lane center is to the vehicle's left -> steer positive (left).
    # Error in pixels:
    e_px = CAMERA_CENTER - cx
    # Converted to metres:
    e_metres = e_px * METRES_PER_PIXEL

    # 2. Stanley Cross-Track Correction:
    # atan2(k * e, v + k_soft)
    crosstrack_correction = math.atan2(K_STANLEY * e_metres, K_SOFT + (DEFAULT_SPEED * 0.05))

    # 3. Total steering angle with heading feedforward and clamping:
    steering = (K_HEADING * psi) + crosstrack_correction
    steering = max(-MAX_STEER, min(MAX_STEER, steering))
    LAST_STATE["steering"] = steering

    # 4. Ackermann geometry split
    left_angle, right_angle = ackermann_wheel_angles(steering)

    return left_angle, right_angle, DEFAULT_SPEED


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
            print(f"Processing Frame: {frames}")

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
