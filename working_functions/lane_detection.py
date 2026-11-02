'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
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
# Author List:      < krish jindal >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
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


def detect_lane(frame):
    '''
    Purpose:
    ---
    Detect the lane in a single frame and report where the centre of the lane
    is, and which of the two lanes the vehicle is currently in.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        A single BGR frame read from the video, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "center_x" : int,   x-pixel of the lane centre in this frame,
                                or -1 if the lane could not be found
            "lane"     : str,   "left", "right" or "unknown"
        }

    Example call:
    ---
    result = detect_lane(frame)

    COORDINATE SYSTEM:
    ---
    `center_x` is an absolute pixel column in the frame AS RECEIVED - the
    dataset's own resolution, 640x480. It is compared against a ground truth
    measured in those pixels, so it only means anything in them.

    You may resize, crop or warp all you like inside this function, but scale
    the answer back before returning it. A centre found in a 320x240 copy is
    half the value it should be, and a centre read off a bird's-eye view is in
    warped coordinates, not frame ones - map the point back through the inverse
    of your transform. Do not re-encode or resize the clip files themselves.

    NOTE:
    ---
    This function must ONLY compute and return the result.
    Do not call cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from
    inside it. All visualisation and debugging output belongs outside this
    function - see draw_overlay() and process_video() below.
    '''



    #################### ADD YOUR CODE HERE ####################
    center_x = -1
    lane = LANE_UNKNOWN

    height, width = frame.shape[:2]
 
    # ---------------------------------------------------------------
    # 1. Isolate yellow and white lane markings.
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
            "lane": LANE_UNKNOWN
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
                "lane": LANE_UNKNOWN
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
                "lane": LANE_UNKNOWN
            }

        lane_center_x = (
            white_x + right_yellow_x
        ) / 2.0

    else:

        return {
            "center_x": -1,
            "lane": LANE_UNKNOWN
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

    return {"center_x": center_x, "lane": lane}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------
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


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
