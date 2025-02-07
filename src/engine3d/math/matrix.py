import numpy as np

from engine3d.math.base.matrixbase import MatrixBase
from engine3d.math.base.vectorbase import VectorBase


class Matrix2x2(MatrixBase):

    dim = (2, 2)

    def __init__(self, *args) -> "Matrix2x2":
        """
        Initialize the matrix.
        """
        super().__init__(*args)


class Matrix3x3(MatrixBase):

    dim = (3, 3)

    def __init__(self, *args) -> "Matrix3x3":
        """
        Initialize the matrix.
        """
        super().__init__(*args)


class Matrix4x4(MatrixBase):

    dim = (4, 4)

    def __init__(self, *args) -> "Matrix3x3":
        """
        Initialize the matrix.
        """
        super().__init__(*args)


class ScaleMatrix2x2Cartesian(Matrix2x2):
    """
    A scale matrix for cartesian coordinates in 2D vectors.
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


class RotationMatrix2x2Cartesian(Matrix2x2):  
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


class TranslationMatrix3x3Homogeneous(Matrix3x3):
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

# TODO: add more matrix types