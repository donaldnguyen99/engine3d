import numpy as np

from engine3d.geometry.base.matrixbase import MatrixBase
from engine3d.geometry.base.vectorbase import VectorBase


class Matrix2D(MatrixBase):
    def __init__(self, *args, **kwargs) -> "Matrix2D":
        """
        Initialize the matrix.
        """
        if "shape" in kwargs:
            super().__init__(np.zeros(kwargs["shape"]))
        super().__init__(*args)


class ScaleMatrix2D(Matrix2D):
    """
    A scale matrix.
    """

    def __init__(self, vector: VectorBase) -> None:
        """
        Initialize the scale matrix.

        Args:
            vector (VectorBase): The vector to scale by.
        """
        super().__init__(np.array([
            [vector.x, 0],
            [0, vector.y]
        ]))

class RotationMatrix2D(Matrix2D):  
    """
    A rotation matrix.
    """

    def __init__(self, angle: float) -> None:
        """
        Initialize the rotation matrix.

        Args:
            angle (float): The angle to rotate by in radians.
        """
        super().__init__(np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]))


class TranslationMatrix2D(Matrix2D):
    """
    A translation matrix.
    """

    def __init__(self, vector: VectorBase) -> None:
        """
        Initialize the translation matrix.

        Args:
            vector (VectorBase): The vector to translate by.
        """
        super().__init__(np.array([
            [1, 0, vector.x],
            [0, 1, vector.y],
            [0, 0, 1]
        ]))


class Matrix3D(MatrixBase):
    def __init__(self, *args, **kwargs) -> "Matrix3D":
        """
        Initialize the matrix.
        """
        if "shape" in kwargs:
            super().__init__(np.zeros(kwargs["shape"]))
        super().__init__(*args)