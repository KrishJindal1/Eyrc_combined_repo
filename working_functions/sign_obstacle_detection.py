'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
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
# Author List:      < Krish >
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import cv2
import numpy as np
import os
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")

# Reference contours are optional until reference glyphs are provided.

MATCH_THRESHOLD = 0.3


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

# def detect_signs(image):
#     '''
#     Purpose:
#     ---
#     Detect every road sign visible in a single image, and report what each
#     one is and how large it appears. An image can hold zero, one, or more
#     than one sign at once.

#     Input Arguments:
#     ---
#     `image` :   [ numpy.ndarray ]
#         A single BGR image, of shape (height, width, 3).

#     Returns:
#     ---
#     `detections` :  [ list of dict ]
#         Zero, one, or many entries - one per sign found in the image:
#         [
#             {
#                 "label" : str,   one of SIGN_LABELS
#                 "area"  : int,   that sign's area in pixels
#             },
#             ...
#         ]

#     Example call:
#     ---
#     detections = detect_signs(image)

#     NOTE:
#     ---
#     This function must ONLY compute and return the result. Do not call
#     cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
#     reading images and reporting results is the evaluator's job, not this
#     function's.
#     '''

#     detections = []

    #################### ADD YOUR CODE HERE ####################
   
def detect_signs(image):
    detections = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red mask
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 80, 50]), np.array([179, 255, 255]))
    )
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    # Blue mask
    blue_mask = cv2.inRange(
        hsv, np.array([85, 50, 30]), np.array([140, 255, 255])
    )
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    blue_contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Detect red School and Speed40 signs
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(contour)

        if not w or not h:
            continue

        ratio = w / float(h)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        if vertices == 3 and 0.65 <= ratio <= 1.35:
            detections.append({
                "label": "School",
                "area": int(area)
            })

        elif 0.80 <= ratio <= 1.20 and circularity >= 0.72:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # A Speed40 sign has a bright, nearly white interior.
            white_mask = cv2.inRange(
                roi_hsv,
                np.array([0, 0, 150]),
                np.array([179, 100, 255])
            )

            # Ignore the outer border.
            mx = max(1, int(w * 0.20))
            my = max(1, int(h * 0.20))
            white_mask[:my, :] = 0
            white_mask[-my:, :] = 0
            white_mask[:, :mx] = 0
            white_mask[:, -mx:] = 0

            white_fraction = cv2.countNonZero(white_mask) / float(w * h)

            if white_fraction >= 0.10:
                detections.append({
                    "label": "Speed40",
                    "area": int(area)
                })
           
    # Find blue sign candidates
    blue_candidates = []

    for contour in blue_contours:
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if not w or not h:
            continue

        ratio = w / float(h)

        if not (0.65 <= ratio <= 1.40):
            continue

        blue_candidates.append((contour, area, x, y, w, h))

    # Remove overlapping duplicate blue candidates
    blue_candidates.sort(key=lambda item: item[1], reverse=True)
    kept = []

    for candidate in blue_candidates:
        _, _, x, y, w, h = candidate
        duplicate = False

        for prior in kept:
            _, _, px, py, pw, ph = prior

            ix1 = max(x, px)
            iy1 = max(y, py)
            ix2 = min(x + w, px + pw)
            iy2 = min(y + h, py + ph)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            smaller = min(w * h, pw * ph)

            if smaller and intersection / smaller > 0.55:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    # Classify Parking and Hospital using reference letters

    # Classify blue signs using their white letter shapes.
    for contour, sign_area, x, y, w, h in kept:
        roi_hsv = hsv[y:y+h, x:x+w]

        white_mask = cv2.inRange(
            roi_hsv,
            np.array([0, 0, 150]),
            np.array([179, 110, 255])
        )

        # Ignore the outer border.
        mx = max(1, int(w * 0.08))
        my = max(1, int(h * 0.08))
        white_mask[:my, :] = 0
        white_mask[-my:, :] = 0
        white_mask[:, :mx] = 0
        white_mask[:, -mx:] = 0

        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        letters, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        letters = [
            c for c in letters
            if cv2.contourArea(c) >= 10
        ]

        if not letters or p_reference is None or h_reference is None:
            continue

        letter = max(letters, key=cv2.contourArea)

        # Normalize the detected and reference contours to comparable sizes.
        def normalize_contour(c):
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw == 0 or bh == 0:
                return None

            shifted = c.copy()
            shifted[:, 0, 0] -= bx
            shifted[:, 0, 1] -= by

            scale = 200.0 / max(bw, bh)
            shifted = np.round(shifted * scale).astype(np.int32)
            return shifted

        detected_shape = normalize_contour(letter)
        p_shape = normalize_contour(p_reference)
        h_shape = normalize_contour(h_reference)

        if detected_shape is None or p_shape is None or h_shape is None:
            continue

        p_score = cv2.matchShapes(
            detected_shape, p_shape, cv2.CONTOURS_MATCH_I1, 0.0
        )
        h_score = cv2.matchShapes(
            detected_shape, h_shape, cv2.CONTOURS_MATCH_I1, 0.0
        )

        best_score = min(p_score, h_score)

        # Reject weak matches rather than labeling every blue region.
        if best_score > MATCH_THRESHOLD:
            continue

        label = "Parking" if p_score < h_score else "Hospital"

        detections.append({
            "label": label,
            "area": int(sign_area)
        })


    return detections


def load_letter_reference(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "reference_images", filename)

    ref_image = cv2.imread(path)
    if ref_image is None:
        return None

    hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)

    # Locate the blue sign, not the white outer background.
    blue_mask = cv2.inRange(
        hsv,
        np.array([85, 50, 30]),
        np.array([140, 255, 255])
    )

    contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    sign = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(sign)

    if w == 0 or h == 0:
        return None

    # Crop slightly inside the blue sign to exclude its border.
    pad_x = int(w * 0.12)
    pad_y = int(h * 0.12)

    x1, y1 = x + pad_x, y + pad_y
    x2, y2 = x + w - pad_x, y + h - pad_y

    roi = ref_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        roi_hsv,
        np.array([0, 0, 150]),
        np.array([179, 110, 255])
    )

    letters, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not letters:
        return None

    # The largest remaining white shape should be the sign's letter.
    letter = max(letters, key=cv2.contourArea)

    if cv2.contourArea(letter) < 10:
        return None

    return letter
p_reference = load_letter_reference("P.jpg")
h_reference = load_letter_reference("Hospital.png")

def detect_obstacle(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 80, 50]), np.array([179, 255, 255]))
    )

    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 300:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if not w or not h:
            continue

        ratio = w / float(h)
        solidity = area / float(w * h)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        approx = cv2.approxPolyDP(
            contour, 0.035 * perimeter, True
        )

        # Reject elongated and circular regions.
        if not (0.65 <= ratio <= 2.3):
            continue

        if solidity < 0.65:
            continue

        if circularity > 0.75:
            continue

        if len(approx) < 4 or len(approx) > 10:
            continue

        if area > best_area:
            best_area = int(area)

    return {
        "present": best_area > 0,
        "area": best_area
    }

  
# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_signs() or detect_obstacle() - the
# evaluation script only ever calls those two functions. List them in
# the file header too.
# ------------------------------------------------------------------



##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area
