"""Separate pedestrian observations from motorcycle riders using box geometry.

This is an inference heuristic, not a pedestrian-trained classification head.
It cannot identify a rider when the motorcycle itself is undetected.
"""

import numpy as np


def pedestrian_labels(names):
    labels = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
    return {index: "pedestrian" if name.lower() == "person" else name
            for index, name in labels.items()}


def filter_motorcycle_riders(xyxy, class_ids, confidences, names, *, motorcycle_confidence):
    """Drop riders before tracking, rendering, and counting; preserve bike boxes.

    A person must overlap at least half its width with the bike, start above
    the bike's midpoint, and end within the bike's vertical span. At least
    20% of the person's area must intersect the bike. Mere proximity does not
    suppress a pedestrian. Weak tracker-only bike candidates cannot suppress
    people. Explicit pedestrian classes retain their trained semantics.
    """
    boxes = np.asarray(xyxy).reshape(-1, 4)
    classes = np.asarray(class_ids)
    scores = np.asarray(confidences)
    labels = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    person_ids = {index for index, name in labels.items() if name.lower() == "person"}
    rider_ids = {index for index, name in labels.items() if name.lower() == "rider"}
    bike_ids = {index for index, name in labels.items() if name.lower() in {"motorcycle", "motorbike", "motor"}}
    keep = ~np.isin(classes, tuple(rider_ids))
    bikes = boxes[np.isin(classes, tuple(bike_ids)) & (scores >= motorcycle_confidence)]
    for index in np.flatnonzero(np.isin(classes, tuple(person_ids))):
        px1, py1, px2, py2 = boxes[index]
        pw, ph = px2 - px1, py2 - py1
        if pw <= 0 or ph <= 0:
            continue
        for bx1, by1, bx2, by2 in bikes:
            if bx2 <= bx1 or by2 <= by1:
                continue
            overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
            overlap_y = max(0, min(py2, by2) - max(py1, by1))
            if (overlap_x / pw >= .5 and overlap_x * overlap_y / (pw * ph) >= .2
                    and py1 < (by1 + by2) / 2 and by1 <= py2 <= by2):
                keep[index] = False
                break
    return boxes[keep], classes[keep], scores[keep], int((~keep).sum())
