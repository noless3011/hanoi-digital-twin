"""Vehicle kinds and their body sizes. Shared by the engine (bodies, contacts) and measurement."""

KINDS = ("motorcycle", "bicycle", "car", "van", "truck", "bus")
# Body length and width (m). Not calibrated.
DIMENSIONS = {"motorcycle": (1.9, 0.75), "bicycle": (1.7, 0.6), "car": (4.4, 1.8), "van": (5.0, 1.95),
              "truck": (7.0, 2.3), "bus": (10.5, 2.5)}
