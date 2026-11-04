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

import warnings

# Camera-view sliding-window lane detector. No perspective transform / BEV.
YELLOW_LOWER = np.array([15, 70, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)
WHITE_LOWER = np.array([0, 0, 170], dtype=np.uint8)
WHITE_UPPER = np.array([180, 90, 255], dtype=np.uint8)

ROI_TOP_RATIO = 0.45
ROI_BOTTOM_RATIO = 0.85

N_WINDOWS = 14
WINDOW_MARGIN = 75
MIN_PIXELS = 12                 # pixels required to add a window to the curve fit
MIN_PIXELS_TO_RECENTER = 5      # lower threshold used only to move the next window
MAX_RECENTER_STEP = 40          # maximum predicted horizontal shift per window
MIN_CURVE_PIXELS = 35
MIN_CURVE_POINTS = 4
MIN_MARKING_SEPARATION = 35

# ---- robust curve fitting (RANSAC) ----------------------------------------
RANSAC_ITERS = 40
RANSAC_MAX_POINTS = 500         # pixels are subsampled to this many for speed
RANSAC_THRESHOLD = 14.0         # px; a pixel this close to the curve is an inlier
QUAD_SPAN_RATIO = 0.35          # points must cover this share of the ROI height
                                # before a 2nd-order fit is trusted (else a line)

# ---- temporal behaviour (uses previous frames) ----------------------------
EMA_ALPHA = 0.6                 # weight of the NEW fit when smoothing coefficients
PRIOR_SEARCH_MARGIN = 70        # px around last frame's curve to seed the search
PRIOR_PEAK_RATIO = 0.35         # prior-seeded histogram peak must be >= this share
                                # of the strongest peak, otherwise search globally
PRIOR_MAX_AGE = 8               # frames a previous curve stays usable as a seed
MAX_HOLD_FRAMES = 5             # frames a lost curve / result may be re-used
                                # (set to 0 to return "unknown" immediately instead)
CURVE_JUMP_PX = 60.0            # mean shift between frames treated as a glitch
JUMP_TOLERANCE_FRAMES = 2       # glitch frames tolerated before believing the change
WIDTH_JUMP_RATIO = 0.35         # allowed change of lane width vs remembered width
WIDTH_TOLERANCE_FRAMES = 5
LANE_HYSTERESIS_PX = 8          # stops the left/right label flickering on the divider
SCENE_CUT_DIFF = 30.0           # mean grey diff (0-255) that resets all memory
LOOKAHEAD_RATIO = 0.10          # look-ahead distance, as a share of image height

# Debug information is stored outside detect_lane's return value. It does not
# affect evaluation and can be inspected by a separate visualisation script.
LAST_DEBUG = {}

# Memory carried from frame to frame (curves, lane width, last good result).
_STATE = {}


def detect_lane(frame):
    """Detect lane centre and lane identity in original camera coordinates."""
    global LAST_DEBUG

    height, width = frame.shape[:2]
    _begin_frame(frame)

    yellow_mask, white_mask = _create_lane_masks(frame)
    road_roi = _create_road_roi(height, width)
    yellow_mask = cv2.bitwise_and(yellow_mask, road_roi)
    white_mask = cv2.bitwise_and(white_mask, road_roi)

    # Remove only broad, bottom-connected vehicle components. A narrow lane
    # marking touching the ROI bottom is deliberately retained.
    white_mask = _remove_vehicle_from_white_mask(white_mask)

    y_top = int(height * ROI_TOP_RATIO)
    y_bottom = int(height * ROI_BOTTOM_RATIO)
    quad_span = QUAD_SPAN_RATIO * (y_bottom - y_top)

    # The reference point is the image centre, as in the submitted approach.
    # The divider's position relative to it determines the current lane.
    reference_y = int(height * 0.63)
    vehicle_x = width / 2.0
    compare_rows = (float(reference_y), (reference_y + y_bottom) / 2.0)

    # 1. Find the white divider first. The previous frame's divider seeds the
    #    search, and its fit is RANSAC-robust, smoothed and glitch-gated.
    white_points, white_windows = _sliding_window_search(
        white_mask, side='full', reference_curve=None,
        prior_curve=_prior_curve('white')
    )
    white_curve, white_state = _temporal_filter(
        'white', _fit_lane_curve(white_points, quad_span), compare_rows
    )

    def fail(reason, left_pts=(), right_pts=(), left_c=None, right_c=None):
        return _fail(frame, yellow_mask, white_mask, white_points,
                     left_pts, right_pts, white_curve, left_c, right_c, reason)

    if white_curve is None:
        return fail("no divider")

    white_x = _evaluate_curve(white_curve, reference_y)
    if white_x is None or not (0 <= white_x < width):
        return fail("divider off image")

    # 2. Current lane (with a little hysteresis), then search ONLY the yellow
    #    boundary on that side of the divider.
    lane = _decide_lane(vehicle_x, white_x)
    side = 'left' if lane == LANE_LEFT else 'right'
    sign = -1.0 if lane == LANE_LEFT else 1.0   # boundary sits at white_x + sign * gap

    boundary_points, boundary_windows = _sliding_window_search(
        yellow_mask, side=side, reference_curve=white_curve,
        prior_curve=_prior_curve(side)
    )
    raw_boundary = _validate_boundary(
        _fit_lane_curve(boundary_points, quad_span),
        white_curve, sign, reference_y, y_bottom, side
    )
    boundary_curve, boundary_state = _temporal_filter(side, raw_boundary, compare_rows)

    left_pts = boundary_points if side == 'left' else []
    right_pts = boundary_points if side == 'right' else []
    left_c = boundary_curve if side == 'left' else None
    right_c = boundary_curve if side == 'right' else None

    boundary_x = (_evaluate_curve(boundary_curve, reference_y)
                  if boundary_curve is not None else None)
    estimated = False
    if boundary_x is None or sign * (boundary_x - white_x) < MIN_MARKING_SEPARATION:
        # Boundary missing or impossible: rebuild it from the remembered lane
        # width instead of giving up (the solid line anchors the lane).
        remembered = _STATE['width'].get(side)
        if remembered is None:
            return fail("boundary missing", left_pts, right_pts, left_c, right_c)
        boundary_x = white_x + sign * remembered
        estimated = True
    elif boundary_state == 'fresh':
        gap = sign * (boundary_x - white_x)
        remembered = _STATE['width'].get(side)
        _STATE['width'][side] = gap if remembered is None else 0.8 * remembered + 0.2 * gap

    lane_center_x = (white_x + boundary_x) / 2.0
    center_x = int(np.clip(round(lane_center_x), 0, width - 1))
    result = {"center_x": center_x, "lane": lane}

    _STATE['lane'] = (lane, _STATE['t'])
    if white_state == 'fresh' and boundary_state == 'fresh' and not estimated:
        _STATE['last_good'] = (dict(result), _STATE['t'])

    # Centre-line curve -> heading, curvature and a look-ahead point. These are
    # kept in LAST_DEBUG because the task's return format is fixed.
    if estimated or boundary_curve is None:
        center_curve = np.array(white_curve, dtype=np.float64)
        center_curve[2] += (boundary_x - white_x) / 2.0
    else:
        center_curve = (white_curve + boundary_curve) / 2.0
    a, b, _ = center_curve
    slope = 2.0 * a * reference_y + b                    # dx/dy at the reference row
    curvature = 2.0 * a / (1.0 + slope ** 2) ** 1.5      # camera-view proxy, not metric
    look_y = max(float(y_top), reference_y - LOOKAHEAD_RATIO * height)
    lookahead_x = float(np.polyval(center_curve, look_y))

    LAST_DEBUG = _make_debug_data(frame, yellow_mask, white_mask,
                                  white_points, left_pts, right_pts,
                                  white_curve, left_c, right_c)
    LAST_DEBUG.update({"reference_y": reference_y, "white_x": white_x,
                       "boundary_x": boundary_x, "lane": lane,
                       "center_x": center_x,
                       "white_windows": white_windows,
                       "left_windows": boundary_windows if side == 'left' else [],
                       "right_windows": boundary_windows if side == 'right' else [],
                       "center_curve": center_curve,
                       "heading_slope": float(slope),
                       "curvature": float(curvature),
                       "lookahead_y": look_y,
                       "lookahead_center_x": lookahead_x,
                       "lookahead_offset_px": lookahead_x - vehicle_x,
                       "white_state": white_state,
                       "boundary_state": boundary_state,
                       "boundary_estimated": estimated})
    return result


# ---------------------------------------------------------------------------
# Memory between frames
# ---------------------------------------------------------------------------

def _reset_state():
    _STATE.clear()
    _STATE.update({'t': 0, 'curves': {}, 'jump': {}, 'width': {}, 'width_bad': 0,
                   'lane': None, 'last_good': None, 'thumb': None, 'shape': None})


def _begin_frame(frame):
    """Advance the frame counter; wipe memory if this frame is not a continuation
    of the previous one (new video / scene cut / different resolution)."""
    if not _STATE:
        _reset_state()
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    previous = _STATE['thumb']
    cut = (previous is None or _STATE['shape'] != frame.shape or
           float(np.mean(np.abs(small - previous))) > SCENE_CUT_DIFF)
    if cut:
        _reset_state()
    _STATE['thumb'] = small
    _STATE['shape'] = frame.shape
    _STATE['t'] += 1


def _prior_curve(key):
    """Last frame's curve for 'white' / 'left' / 'right', if recent enough."""
    entry = _STATE['curves'].get(key)
    if entry is None or _STATE['t'] - entry[1] > PRIOR_MAX_AGE:
        return None
    return entry[0]


def _temporal_filter(key, new_curve, rows):
    """Smooth a new fit against the previous one.

    Returns (curve, status) with status 'fresh' (new fit used), 'held' (previous
    curve re-used because the new fit was missing or a sudden jump) or 'none'.
    """
    t = _STATE['t']
    entry = _STATE['curves'].get(key)
    age = None if entry is None else t - entry[1]
    can_hold = entry is not None and age <= MAX_HOLD_FRAMES
    previous = entry[0] if entry is not None and age <= PRIOR_MAX_AGE else None

    if new_curve is None:
        return (entry[0], 'held') if can_hold else (None, 'none')

    if can_hold and _curve_jump(new_curve, entry[0], rows) > CURVE_JUMP_PX:
        count = _STATE['jump'].get(key, 0) + 1
        if count <= JUMP_TOLERANCE_FRAMES:
            _STATE['jump'][key] = count
            return entry[0], 'held'
        smoothed = new_curve            # change persisted: believe it
    elif previous is None:
        smoothed = new_curve
    else:
        smoothed = EMA_ALPHA * new_curve + (1.0 - EMA_ALPHA) * previous

    _STATE['jump'][key] = 0
    _STATE['curves'][key] = (smoothed, t)
    return smoothed, 'fresh'


def _curve_jump(curve_a, curve_b, rows):
    ys = np.asarray(rows, dtype=np.float64)
    return float(np.mean(np.abs(np.polyval(curve_a, ys) - np.polyval(curve_b, ys))))


def _decide_lane(vehicle_x, white_x):
    """Left/right of the divider, sticking to the previous label near the line."""
    remembered = _STATE.get('lane')
    if remembered is not None and _STATE['t'] - remembered[1] <= PRIOR_MAX_AGE:
        if remembered[0] == LANE_LEFT and vehicle_x < white_x + LANE_HYSTERESIS_PX:
            return LANE_LEFT
        if remembered[0] == LANE_RIGHT and vehicle_x >= white_x - LANE_HYSTERESIS_PX:
            return LANE_RIGHT
    return LANE_LEFT if vehicle_x < white_x else LANE_RIGHT


def _validate_boundary(curve, white_curve, sign, reference_y, y_near, side):
    """Sanity-check a fresh yellow fit against the divider and the lane width
    seen before. Returns the curve, or None if it should be distrusted."""
    if curve is None:
        return None
    w_ref = _evaluate_curve(white_curve, reference_y)
    b_ref = _evaluate_curve(curve, reference_y)
    w_near = _evaluate_curve(white_curve, y_near)
    b_near = _evaluate_curve(curve, y_near)
    if None in (w_ref, b_ref, w_near, b_near):
        return None
    gap_ref = sign * (b_ref - w_ref)
    gap_near = sign * (b_near - w_near)
    # Correct side of the divider at two rows (no crossing / mirrored fit).
    if gap_ref < MIN_MARKING_SEPARATION or gap_near < 0.5 * MIN_MARKING_SEPARATION:
        return None
    remembered = _STATE['width'].get(side)
    if remembered is not None and abs(gap_ref - remembered) > WIDTH_JUMP_RATIO * remembered:
        _STATE['width_bad'] += 1
        if _STATE['width_bad'] <= WIDTH_TOLERANCE_FRAMES:
            return None
        _STATE['width'].pop(side, None)   # persistent change: re-learn the width
    _STATE['width_bad'] = 0
    return curve


def _fail(frame, yellow_mask, white_mask, white_points, left_pts, right_pts,
          white_curve, left_curve, right_curve, reason):
    """Record debug data and return the last good result (briefly) or unknown."""
    global LAST_DEBUG
    LAST_DEBUG = _make_debug_data(frame, yellow_mask, white_mask, white_points,
                                  left_pts, right_pts, white_curve,
                                  left_curve, right_curve)
    LAST_DEBUG["fail_reason"] = reason
    held = _STATE.get('last_good')
    if held is not None and _STATE['t'] - held[1] <= MAX_HOLD_FRAMES:
        LAST_DEBUG["held_last_good"] = True
        return dict(held[0])
    return {"center_x": -1, "lane": LANE_UNKNOWN}


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _create_lane_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    return yellow, white


def _create_road_roi(height, width):
    """Use a rectangular camera-view ROI; no image warp is performed."""
    roi = np.zeros((height, width), dtype=np.uint8)
    top_y = int(height * ROI_TOP_RATIO)
    bottom_y = int(height * ROI_BOTTOM_RATIO)
    roi[top_y:bottom_y, :] = 255
    return roi


def _sliding_window_search(mask, side='full', reference_curve=None, prior_curve=None):
    """Track marking pixels bottom-to-top with adaptive sliding windows.

    A window with a small but usable number of pixels can recenter the next
    window even when it does not contain enough pixels to contribute to the
    final curve fit. If a window is empty, the next centre is predicted from
    the movement observed in the last two successful windows - or, if there is
    no history yet, from last frame's curve (`prior_curve`).

    `prior_curve` also seeds the starting column: the histogram peak nearest to
    where the marking was last frame wins over a stronger but distant peak.
    """
    height, width = mask.shape
    y_bottom = int(height * ROI_BOTTOM_RATIO)
    y_top = int(height * ROI_TOP_RATIO)
    window_height = max(1, (y_bottom - y_top) // N_WINDOWS)
    nonzero_y, nonzero_x = mask.nonzero()

    # Initialize from the lower part of the ROI.
    hist_y_start = y_top + int((y_bottom - y_top) * 0.55)
    bottom_mid_y = (hist_y_start + y_bottom) / 2.0
    xs = np.arange(width)

    if reference_curve is None:
        search_region = np.ones(width, dtype=bool)
    else:
        divider_bottom_x = float(np.polyval(reference_curve, bottom_mid_y))
        if side == 'left':
            search_region = xs < divider_bottom_x - MIN_MARKING_SEPARATION
        elif side == 'right':
            search_region = xs > divider_bottom_x + MIN_MARKING_SEPARATION
        else:
            search_region = np.ones(width, dtype=bool)

    def restricted_histogram(y_start, y_end):
        hist = np.sum(mask[y_start:y_end, :] > 0, axis=0).astype(np.float64)
        hist = np.convolve(hist, np.ones(9, dtype=np.float64) / 9.0, mode='same')
        hist[~search_region] = 0
        return hist

    prior_x = None
    if prior_curve is not None:
        value = float(np.polyval(prior_curve, bottom_mid_y))
        if np.isfinite(value):
            prior_x = value

    restricted_hist = restricted_histogram(hist_y_start, y_bottom)
    peak = restricted_hist.max(initial=0)
    if peak > 0:
        current_x = int(np.argmax(restricted_hist))
        if prior_x is not None:
            lo = max(0, int(prior_x) - PRIOR_SEARCH_MARGIN)
            hi = min(width, int(prior_x) + PRIOR_SEARCH_MARGIN + 1)
            local = restricted_hist[lo:hi]
            if local.size and local.max() > 0 and local.max() >= PRIOR_PEAK_RATIO * peak:
                current_x = lo + int(np.argmax(local))
    elif prior_x is not None:
        # Dash gap across the whole lower ROI: trust last frame's position.
        current_x = int(np.clip(prior_x, 0, width - 1))
    else:
        # Dash gap and no history: look over the full ROI height instead.
        wide = restricted_histogram(y_top, y_bottom)
        if wide.max(initial=0) <= 0:
            return [], []
        current_x = int(np.argmax(wide))

    points_x, points_y = [], []
    windows = []
    # (window index, detected centre) pairs used to estimate curve direction.
    centre_history = []

    for window_index in range(N_WINDOWS):
        y_high = y_bottom - window_index * window_height
        y_low = max(y_top, y_bottom - (window_index + 1) * window_height)
        y_mid = (y_low + y_high) / 2.0

        if reference_curve is not None:
            divider_x = float(np.polyval(reference_curve, y_mid))
            if side == 'left':
                x_low_limit, x_high_limit = 0, int(divider_x - MIN_MARKING_SEPARATION)
            elif side == 'right':
                x_low_limit, x_high_limit = int(divider_x + MIN_MARKING_SEPARATION), width
            else:
                x_low_limit, x_high_limit = 0, width
            x_low_limit = max(0, x_low_limit)
            x_high_limit = min(width, x_high_limit)
        else:
            x_low_limit, x_high_limit = 0, width

        # Keep the predicted centre inside the valid side of the divider.
        if x_high_limit > x_low_limit:
            current_x = int(np.clip(current_x, x_low_limit, x_high_limit - 1))

        x_low = max(x_low_limit, current_x - WINDOW_MARGIN)
        x_high = min(x_high_limit, current_x + WINDOW_MARGIN + 1)
        windows.append((x_low, y_low, x_high, y_high))
        if x_high <= x_low:
            continue

        good = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high))
        good_x = nonzero_x[good]
        good_y = nonzero_y[good]
        pixel_count = len(good_x)

        # Only sufficiently populated windows contribute pixels to the fit.
        if pixel_count >= MIN_PIXELS:
            points_x.extend(good_x.tolist())
            points_y.extend(good_y.tolist())

        # Recenter at a lower threshold than the curve-fit threshold.
        # This prevents sparse marking fragments from freezing the tracker.
        if pixel_count >= MIN_PIXELS_TO_RECENTER:
            detected_x = float(np.median(good_x))
            # For sparse windows, blend the measurement with the trajectory
            # prediction so a few stray pixels cannot cause a large jump.
            if len(centre_history) >= 2:
                (prev_i, prev_x), (last_i, last_x) = centre_history[-2:]
                steps = max(1, last_i - prev_i)
                slope = (last_x - prev_x) / float(steps)
                slope = float(np.clip(slope, -MAX_RECENTER_STEP, MAX_RECENTER_STEP))
                predicted_x = last_x + slope * (window_index - last_i)
            elif centre_history:
                predicted_x = centre_history[-1][1]
            else:
                predicted_x = float(current_x)

            if pixel_count < MIN_PIXELS:
                detected_x = 0.65 * detected_x + 0.35 * predicted_x

            current_x = int(round(detected_x))
            centre_history.append((window_index, current_x))
        elif len(centre_history) >= 2:
            # No usable pixels: continue along the recent trajectory instead
            # of repeatedly searching around a stale centre.
            (prev_i, prev_x), (last_i, last_x) = centre_history[-2:]
            steps = max(1, last_i - prev_i)
            slope = (last_x - prev_x) / float(steps)
            slope = float(np.clip(slope, -MAX_RECENTER_STEP, MAX_RECENTER_STEP))
            current_x = int(round(last_x + slope * (window_index - last_i)))
        elif centre_history:
            current_x = centre_history[-1][1]
        elif prior_curve is not None:
            # Nothing seen yet (start of a dash gap): follow last frame's curve.
            value = float(np.polyval(prior_curve, y_mid))
            if np.isfinite(value):
                current_x = int(round(value))

    if len(points_x) < MIN_CURVE_PIXELS:
        return [], windows
    points = list(zip(np.asarray(points_x, dtype=float),
                      np.asarray(points_y, dtype=float)))
    return points, windows


def _ransac_inliers(y_values, x_values, degree):
    """RANSAC for x = f(y). Each trial draws one pixel from each vertical
    stratum, so a trial can never be a degenerate cluster in a single row."""
    count = len(y_values)
    rng = np.random.RandomState(0)          # fixed seed -> reproducible results
    if count > RANSAC_MAX_POINTS:
        keep = rng.choice(count, RANSAC_MAX_POINTS, replace=False)
    else:
        keep = np.arange(count)
    ys, xs = y_values[keep], x_values[keep]
    strata = [s for s in np.array_split(np.argsort(ys), degree + 1) if len(s)]
    if len(strata) < degree + 1:
        return None

    best_curve, best_score = None, -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(RANSAC_ITERS):
            sample = np.array([rng.choice(s) for s in strata])
            try:
                curve = np.polyfit(ys[sample], xs[sample], degree)
            except (np.linalg.LinAlgError, ValueError):
                continue
            score = int(np.count_nonzero(
                np.abs(xs - np.polyval(curve, ys)) <= RANSAC_THRESHOLD))
            if score > best_score:
                best_curve, best_score = curve, score
    if best_curve is None:
        return None
    return np.abs(x_values - np.polyval(best_curve, y_values)) <= RANSAC_THRESHOLD


def _fit_lane_curve(points, quad_span=0.0):
    """Robust x = a*y^2 + b*y + c fit (always returns 3 coefficients).

    Outliers are removed by RANSAC, then the curve is refitted on the inliers.
    If the pixels span less than `quad_span` rows (e.g. a single dash), a
    straight line is fitted instead: a quadratic through one short dash would
    extrapolate wildly to the reference row.
    """
    if points is None or len(points) < MIN_CURVE_PIXELS:
        return None
    data = np.asarray(points, dtype=np.float64)
    x_values, y_values = data[:, 0], data[:, 1]
    degree = 2 if (y_values.max() - y_values.min()) >= quad_span else 1
    try:
        inliers = _ransac_inliers(y_values, x_values, degree)
        if inliers is None or np.count_nonzero(inliers) < MIN_CURVE_PIXELS:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = np.polyfit(y_values[inliers], x_values[inliers], degree)
            errors = np.abs(x_values - np.polyval(curve, y_values))
            threshold = max(RANSAC_THRESHOLD, 2.5 * float(np.median(errors[inliers])))
            refined = errors <= threshold
            if np.count_nonzero(refined) >= MIN_CURVE_PIXELS:
                curve = np.polyfit(y_values[refined], x_values[refined], degree)
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None
    if degree == 1:
        curve = np.array([0.0, curve[0], curve[1]])
    if not np.all(np.isfinite(curve)):
        return None
    return curve


def _evaluate_curve(curve, y):
    if curve is None:
        return None
    value = float(np.polyval(curve, y))
    return value if np.isfinite(value) else None


def _remove_vehicle_from_white_mask(mask):
    """Remove broad vehicle-body blobs, not narrow lane markings."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape
    for label in range(1, count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        touches_bottom = (y + component_height >= height - 1)
        overlaps_vehicle_region = (x < int(width * 0.75) and
                                   x + component_width > int(width * 0.25))
        if touches_bottom and overlaps_vehicle_region and area > 500:
            mask[labels == label] = 0
    return mask


def _make_debug_data(frame, yellow_mask, white_mask, white_points,
                     left_points, right_points, white_curve, left_curve, right_curve):
    return {
        "frame": frame.copy(),
        "yellow_mask": yellow_mask.copy(),
        "white_mask": white_mask.copy(),
        "white_points": white_points,
        "left_yellow_points": left_points,
        "right_yellow_points": right_points,
        "white_curve": white_curve,
        "left_curve": left_curve,
        "right_curve": right_curve,
    }


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