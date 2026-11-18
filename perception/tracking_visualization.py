"""Draw observed identities, trails and missed tracks beside a video frame."""

from __future__ import annotations

import math


PANEL_WIDTH = 300
MIN_HEIGHT = 420
BACKGROUND = (24, 20, 17)
TEXT = (238, 236, 230)
MUTED = (162, 154, 142)
ACCENT = (205, 219, 78)
COLORS = (
    (100, 230, 255), (255, 170, 100), (160, 255, 130), (220, 135, 255),
    (110, 165, 255), (245, 235, 100), (180, 130, 255), (140, 255, 215),
    (255, 160, 215), (185, 220, 130), (110, 215, 245), (235, 185, 155),
)


def track_color(track_id):
    """Colors belong to identities, so class changes cannot change the color."""
    return COLORS[(int(track_id) - 1) % len(COLORS)]


def _text(image, value, point, cv2, color=TEXT, scale=.48):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _label(image, value, box, color, cv2, *, lost=False):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    scale = max(.25, .46 * min(1, width / 640, height / 480))
    padding = max(1, round(3 * scale / .46))
    # Keep each label within half the feed width, even for long class names.
    max_width = max(0, width // 2 - 2 * padding)
    measure = lambda text: cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    if measure(value)[0] > max_width:
        shortened = value
        while shortened and measure(shortened + "...")[0] > max_width:
            shortened = shortened[:-1]
        value = shortened + "..." if shortened else ""
    text_width, text_height = measure(value)
    if not value or text_height + 2 * padding >= height:
        return
    x = max(0, min(x1, width - text_width - 2 * padding - 1))
    y = max(text_height + padding, min(height - padding - 1, y1 - padding))
    fill = tuple(int(channel * .35) for channel in color) if lost else color
    cv2.rectangle(image, (x, y - text_height - padding),
                  (x + text_width + 2 * padding, y + padding), fill, -1)
    _text(image, value, (x + padding, y), cv2, TEXT if lost else (18, 18, 18), scale)


def _dashed_box(image, box, color, cv2):
    x1, y1, x2, y2 = box
    for start in range(x1, x2, 14):
        end = min(start + 7, x2)
        cv2.line(image, (start, y1), (end, y1), color, 1, cv2.LINE_AA)
        cv2.line(image, (start, y2), (end, y2), color, 1, cv2.LINE_AA)
    for start in range(y1, y2, 14):
        end = min(start + 7, y2)
        cv2.line(image, (x1, start), (x1, end), color, 1, cv2.LINE_AA)
        cv2.line(image, (x2, start), (x2, end), color, 1, cv2.LINE_AA)


def annotate_tracking(frame, xyxy, class_id, confidence, labels, snapshot,
                      inference_ms, tracking_ms, cv2, *, detector="RF-DETR",
                      dropped_frames=0, display_confidence=0.0):
    """Append a diagnostics panel; missed tracks stay at their last observation.

    A None snapshot explicitly means detection only. The source pixels and
    detection arrays are never modified. FPS here is detector inference only.
    """
    import numpy as np

    boxes = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    classes = np.asarray(class_id).reshape(-1)
    scores = np.asarray(confidence, dtype=float).reshape(-1)
    if len(boxes) != len(classes) or len(boxes) != len(scores):
        raise ValueError("Detection boxes, classes and confidences must have equal lengths.")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite boxes or confidence scores.")
    if any(not math.isfinite(float(cls)) or float(cls) != int(cls) or int(cls) not in labels for cls in classes):
        raise ValueError("Model returned a class index absent from its class mapping.")
    height, width = frame.shape[:2]
    annotated = np.full((max(MIN_HEIGHT, height), width + PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    annotated[:height, :width] = frame
    video = annotated[:height, :width]
    compact_labels = width < 640 or height < 360
    detected = unassigned = 0
    for index, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1 or score < display_confidence:
            continue
        detected += 1
        if snapshot is not None and snapshot.detection_ids[index] is not None:
            continue
        unassigned += 1
        color = (160, 160, 160)
        cv2.rectangle(video, (x1, y1), (x2, y2), color, 1)
        suffix = "unassigned" if snapshot is not None else "detection"
        value = labels[int(cls)] if compact_labels else f"{labels[int(cls)]} {score:.2f} | {suffix}"
        _label(video, value, (x1, y1, x2, y2), color, cv2)

    if snapshot is not None:
        # Lost tracks first, so a current observation remains visually dominant.
        for track in sorted(snapshot.tracks, key=lambda item: item.missed == 0):
            color = track_color(track.track_id)
            faded = tuple(int(channel * .55) for channel in color)
            x1, y1, x2, y2 = (int(round(value)) for value in track.box)
            box = (max(0, min(width - 1, x1)), max(0, min(height - 1, y1)),
                   max(0, min(width - 1, x2)), max(0, min(height - 1, y2)))
            trail = [(max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y)))))
                     for x, y in track.trail]
            for index in range(1, len(trail)):
                intensity = .25 + .75 * index / max(1, len(trail) - 1)
                trail_color = faded if track.missed else tuple(int(channel * intensity) for channel in color)
                cv2.line(video, trail[index - 1], trail[index], trail_color, 2, cv2.LINE_AA)
            if trail:
                cv2.circle(video, trail[-1], 3, faded if track.missed else color, -1, cv2.LINE_AA)
            if not track.missed and len(trail) >= 2 and trail[-1] != trail[-2]:
                cv2.arrowedLine(video, trail[-2], trail[-1], color, 2, cv2.LINE_AA, tipLength=.35)
            name = labels.get(track.class_id, str(track.class_id))
            if track.missed:
                _dashed_box(video, box, faded, cv2)
                value = f"#{track.track_id} {name} | LOST +{track.missed}"
            else:
                cv2.rectangle(video, box[:2], box[2:], color, 2)
                value = f"#{track.track_id} {name} {track.confidence:.2f} | {track.state}"
            if compact_labels:
                value = f"#{track.track_id} {name}" + (" LOST" if track.missed else "")
            _label(video, value, box, color, cv2, lost=bool(track.missed))

    title = f"{detector} + TrackTrack" if snapshot is not None else f"{detector} | DETECTION ONLY"
    # Use the existing space below short feeds instead of covering source pixels.
    header_y = height if annotated.shape[0] - height >= 32 else 0
    header = annotated[header_y:header_y + 32, :width]
    cv2.rectangle(header, (0, 0), (width - 1, min(30, header.shape[0] - 1)), BACKGROUND, -1)
    title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0]
    title_scale = .55 * min(1, max(1, width - 20) / max(1, title_width))
    _text(header, title, (min(10, width - 1), min(21, header.shape[0] - 1)), cv2, TEXT, title_scale)

    panel = annotated[:, width:]
    panel_height = panel.shape[0]
    enabled = snapshot is not None
    _text(panel, "TRACKING LIVE" if enabled else "DETECTION ONLY", (16, 26), cv2, ACCENT if enabled else (120, 185, 255), .64)
    _text(panel, "TrackTrack | appearance matching OFF" if enabled else "No persistent identities or trails", (16, 46), cv2, MUTED, .40)
    cv2.line(panel, (16, 57), (PANEL_WIDTH - 16, 57), (62, 56, 49), 1)
    if enabled:
        _text(panel, f"VISIBLE  {snapshot.visible:>3}     LOST  {snapshot.lost:>3}", (16, 82), cv2, TEXT, .54)
        _text(panel, f"NEW  {snapshot.new:>3}     RECOVERED  {snapshot.recovered:>3}", (16, 106), cv2, TEXT, .46)
        _text(panel, f"IDs created this session: {snapshot.total_ids}", (16, 130), cv2, ACCENT, .46)
    else:
        _text(panel, f"DETECTIONS  {detected}", (16, 82), cv2, TEXT, .54)
        _text(panel, "Enable tracking to inspect continuity", (16, 106), cv2, MUTED, .40)
    _text(panel, f"Detections: {detected}   Unassigned: {unassigned if enabled else '-'}", (16, 152), cv2, MUTED, .43)
    fps = 1000 / inference_ms if inference_ms > 0 else 0
    _text(panel, f"Detector {inference_ms:.0f} ms | {fps:.1f} infer FPS", (16, 174), cv2, MUTED, .43)
    _text(panel, f"Tracker {tracking_ms:.1f} ms | Skipped frames: {dropped_frames}", (16, 194), cv2, MUTED, .41)
    cv2.line(panel, (16, 206), (PANEL_WIDTH - 16, 206), (62, 56, 49), 1)
    _text(panel, "Solid = observed | Dashed = last seen", (16, 225), cv2, MUTED, .41)
    _text(panel, "Same ID + color should follow one object", (16, 243), cv2, MUTED, .39)
    _text(panel, "ID / CLASS                STATE / HITS", (16, 270), cv2, TEXT, .40)
    capacity = max(1, min(12, (panel_height - 328) // 37))
    tracks = sorted(snapshot.tracks, key=lambda item: (item.missed > 0, -item.hits, item.track_id)) if enabled else []
    for index, track in enumerate(tracks[:capacity]):
        y = 292 + index * 37
        color = track_color(track.track_id)
        cv2.rectangle(panel, (16, y - 10), (21, y + 9), color, -1)
        name = labels.get(track.class_id, str(track.class_id))
        _text(panel, f"#{track.track_id} {name[:15]}", (29, y), cv2, color, .46)
        _text(panel, track.state, (190, y), cv2, color if not track.missed else MUTED, .37)
        _text(panel, f"{track.hits} observations | missed {track.missed}", (29, y + 15), cv2, MUTED, .37)
    if not tracks:
        _text(panel, "Waiting for object tracks..." if enabled else "Restart without --no-tracking", (16, 295), cv2, MUTED, .42)
    elif len(tracks) > capacity:
        _text(panel, f"+ {len(tracks) - capacity} more tracks", (16, panel_height - 55), cv2, MUTED, .40)
    _text(panel, "IDs are tracker hypotheses, not verified counts", (16, panel_height - 33), cv2, MUTED, .35)
    _text(panel, "NEW = tentative | RECOVERED = seen again", (16, panel_height - 17), cv2, MUTED, .36)
    return annotated


def annotate_waiting(image, cv2):
    """Make a stale preview's paused status visible across its entire header."""
    waiting = image.copy()
    cv2.rectangle(waiting, (0, 0), (waiting.shape[1] - 1, 32), BACKGROUND, -1)
    _text(waiting, "WAITING FOR NEW FRAMES | processing paused", (10, 23), cv2, (120, 185, 255), .55)
    return waiting
