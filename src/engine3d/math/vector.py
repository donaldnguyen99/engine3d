from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from engine3d.math.base.vectorbase import VectorBase
from engine3d.math.matrix import \
    Matrix4x4,\
    RotationMatrix3x3Homogeneous, RotationMatrix4x4Homogeneous


class Vector2D(VectorBase):

    dim = 2

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)
    
    def __mul__(self, other):
        res = super().__mul__(other)
        if res == NotImplemented:
            raise NotImplementedError(f"* operation between Vector2D and {type(other)} \
                             is not implemented. * must be followed by a scalar")
        return res

    def rotate(self, angle: float) -> "VectorBase":
        """
        Rotates a vector by an angle, inplace.

        Args:
            angle (float): The angle to rotate by.

        Returns:
            VectorBase: The vector object rotated inplace.
        """
        self.array = self.rotated(angle).array
        return self
    
    def rotated(self, angle: float) -> "VectorBase":
        """
        Rotates a vector by an angle.
        
        Args:
            angle (float): The angle to rotate by.
        
        Returns:
            VectorBase: A new rotated vector.
        """
        return Vector2D(
            self.x * np.cos(angle) - self.y * np.sin(angle), 
            self.x * np.sin(angle) + self.y * np.cos(angle))
    
    def scale(self, scale: float) -> "VectorBase":
        self.array *= scale
        return self
    
    def translate(self, translation: "VectorBase") -> "VectorBase":
        self.array += translation.array
        return self
    

class Vector3D(VectorBase):

    dim = 3

    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)
    
    
    def __mul__(self, other):
        res = super().__mul__(other)
        if res == NotImplemented:
            raise NotImplementedError(f"* operation between Vector3D and {type(other)} \
                             is not implemented. * must be followed by a scalar")
        return res


    def rotate(self, angle: float, axis: "VectorBase") -> "VectorBase":
        self.array = self.rotated(angle, axis).array
        return self

    def rotated(self, angle: float, axis: "VectorBase") -> "VectorBase":
        return -self.cross(axis) * np.sin(angle) + self * np.cos(angle) + axis * self.dot(axis) * (1 - np.cos(angle))
    
    def rotate3d(self, angle: float, axis: "VectorBase") -> "VectorBase":
        return self.rotate(angle, axis)
    
    def rotated3d(self, angle: float, axis: "VectorBase") -> "VectorBase":
        return self.rotated(angle, axis)
    
    def rotate2d(self, angle: float) -> "VectorBase":
        self.array = (RotationMatrix3x3Homogeneous(angle) @ self).array
        return self
    
    def rotated2d(self, angle: float) -> "VectorBase":
        return RotationMatrix3x3Homogeneous(angle) @ self

    def orthonormal_basis(self) -> tuple[VectorBase, VectorBase, VectorBase]:
        """
        Returns an orthonormal basis given a single vector.

        Args:
            v1 (VectorBase): The first vector.

        Returns:
            tuple[VectorBase, VectorBase, VectorBase]: An orthonormal basis.
        """
        v1_normalized = self.normalized()
        axis1 = Vector3D(1, 0, 0)
        if abs(abs(v1_normalized.dot(axis1)) - 1) < self.EPSILON:
            axis1 = Vector3D(0, 0, 1)
        v2 = v1_normalized.rotated(np.pi / 2, axis1)
        v3 = v1_normalized.cross(v2)
        return v1_normalized, v2, v3

    def azimuth_elevation_between(self, other: "Vector3D") -> tuple[float, float]:
        """
        Returns the azimuth and elevation between two vectors. A positive azimuth 
        means the other vector is counter-clockwise about the z-axis to the current 
        vector. A positive elevation means the other vector is above the current vector.

        Args:
            other (Vector3D): The other vector.

        Returns:
            tuple[float, float]: The azimuth and elevation.
        """
        azimuth = np.arctan2(self.xy.cross(other.xy), self.xy.dot(other.xy))
        v = Vector2D(other.z, other.xy.magnitude)
        u = Vector2D(self.z, self.xy.magnitude)
        elevation = np.arctan2(-u.cross(v), u.dot(v)) # Need to optimize
        return azimuth, elevation
    
    def rotate_by_azimuth_elevation(self, azimuth: float, elevation: float) -> "Vector3D":
        """
        Rotates a vector by a change of azimuth angle and elevation.

        Args:
            azimuth (float): The azimuth angle.
            elevation (float): The elevation angle.

        Returns:
            VectorBase: The rotated vector.
        """
        if abs(abs(self.normalized().dot(Vector3D(0, 0, 1))) - 1) < self.EPSILON:
            if abs(elevation) > self.EPSILON:
                raise ValueError("The vector is parallel to the z-axis.")
            else:
                return self
        axis1 = self.cross(Vector3D(0, 0, 1))
        return self.rotated(elevation, axis1).rotated(azimuth, Vector3D(0, 0, 1))
    
    def scale3d(self, scale: float) -> "Vector3D":
        self.array *= scale
        return self
    
    def scale2d(self, scale: float) -> "Vector2D":
        self.array[:2] *= scale
        return self
    
    def cartesian_to_homogeneous(self) -> "Vector4D":
        return Vector4D(self.x, self.y, self.z, 1)

class Vector4D(VectorBase):

    dim = 4

    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        super().__init__(x, y, z, w)
    
    def __mul__(self, other):
        res = super().__mul__(other)
        if res == NotImplemented:
            raise NotImplementedError(f"* operation between Vector3D and {type(other)} \
                             is not implemented. * must be followed by a scalar")
        return res
    
    def rotate(self, *args):
        self.rotate3d(*args)
        return self

    def rotated(self, *args):
        return self.rotated3d(*args)

    def rotate3d(self, angle: float, axis: "VectorBase") -> "VectorBase":
        self.array = (RotationMatrix4x4Homogeneous(angle, axis) @ self).array
        return self
    
    def rotated3d(self, angle: float, axis: "VectorBase") -> "VectorBase":
        return RotationMatrix4x4Homogeneous(angle, axis) @ self
    
    def transform_matrix(self, matrix: "Matrix4x4") -> "Vector4D":
        self.array = (matrix @ self).array
        return self
    
    def homogeneous_to_cartesian(self) -> "Vector3D":
        return Vector3D(self.x / self.w, self.y / self.w, self.z / self.w)