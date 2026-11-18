"""Hanoi traffic digital twin: a light, deterministic 2D simulation core, measurement from video
trajectories, and calibration. Unity only plays results back.

Parallel runs use one process per core, so numeric libraries are kept to one thread each.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
