import numpy as np

from engine3d.math.base.matrixbase import MatrixBase
from engine3d.math.base.vectorbase import VectorBase


class Matrix2x2(MatrixBase):

    dim = (2, 2)

    def __init__(self, *args) -> "Matrix2x2":
        super().__init__(*args)


class Matrix3x3(MatrixBase):

    dim = (3, 3)

    def __init__(self, *args) -> "Matrix3x3":
        super().__init__(*args)


class Matrix4x4(MatrixBase):

    dim = (4, 4)

    def __init__(self, *args) -> "Matrix3x3":
        super().__init__(*args)
    
    def __matmul__(self, other):
        if isinstance(other, Matrix4x4):
            return Matrix4x4(self.array @ other.array)
        elif isinstance(other, VectorBase):
            return other.__class__(*(self.array @ other.array))
        raise TypeError(f"Cannot matrix multiply {self.__class__.__name__} by {other.__class__.__name__} using @." + (" Use * instead." if isinstance(other, (int, float)) else ""))



class ScaleMatrix2x2Cartesian(Matrix2x2):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [vector.x, 0],
            [0, vector.y]
        ]))


class RotationMatrix2x2Cartesian(Matrix2x2):  
    def __init__(self, angle: float) -> None:
        super().__init__(np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]))


class TranslationMatrix3x3Homogeneous(Matrix3x3):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [1, 0, vector.x],
            [0, 1, vector.y],
            [0, 0, 1]
        ]))

class ScaleMatrix3x3Homogeneous(Matrix3x3):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [vector.x, 0, 0],
            [0, vector.y, 0],
            [0, 0, 1]
        ]))

class RotationMatrix3x3Homogeneous(Matrix3x3):
    def __init__(self, angle: float) -> None:
        super().__init__(np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ]))

class ShearMatrix3x3Homogeneous(Matrix3x3):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [1, vector.x, 0],
            [vector.y, 1, 0],
            [0, 0, 1]
        ]))

class ScaleMatrix3x3Cartesian(Matrix3x3):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [vector.x, 0, 0],
            [0, vector.y, 0],
            [0, 0, vector.z]
        ]))

class RotationMatrix3x3Cartesian(Matrix3x3):
    def __init__(self, angle: float, axis: VectorBase) -> None:
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        axis.normalize()
        x = axis.x
        y = axis.y
        z = axis.z
        super().__init__(np.array([
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ]))

class TranslationMatrix4x4Homogeneous(Matrix4x4):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [1, 0, 0, vector.x],
            [0, 1, 0, vector.y],
            [0, 0, 1, vector.z],
            [0, 0, 0, 1]
        ]))

class RotationMatrix4x4Homogeneous(Matrix4x4):
    def __init__(self, angle: float, axis: VectorBase) -> None:
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        axis.normalize()
        x = axis.x
        y = axis.y
        z = axis.z
        super().__init__(np.array([
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y, 0],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x, 0],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c, 0],
            [0, 0, 0, 1]
        ]))

class ScaleMatrix4x4Homogeneous(Matrix4x4):
    def __init__(self, vector: VectorBase) -> None:
        super().__init__(np.array([
            [vector.x, 0, 0, 0],
            [0, vector.y, 0, 0],
            [0, 0, vector.z, 0],
            [0, 0, 0, 1]
        ]))

class PerspectiveProjectionMatrix4x4Homogeneous(Matrix4x4):
    def __init__(self, fov: float, aspect: float, near: float, far: float) -> None:
        """
        Perspective projection matrix
        Args:
            fov: field of view in radians
            aspect: aspect ratio (width / height)
            near: near plane
            far: far plane
        """
        assert fov > 0 and fov < np.pi
        assert aspect > 0
        assert near > 0
        assert far > near
        super().__init__(np.array([
            [1 / (aspect * np.tan(fov / 2)), 0, 0, 0],
            [0, 1 / np.tan(fov / 2), 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0]
        ]))


        