import numpy as np

from engine3d.math.vector import Vector3D


class Object3D:

    vertices: np.ndarray
    faces: np.ndarray

    def __init__(self) -> None:
        self.vertices = np.array([
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1)
        ])
        self.faces = np.array([
            (0, 1, 3, 2),
            (4, 5, 7, 6),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 3, 7, 5)
        ])

