from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from engine3d.geometry.base.vectorbase import VectorBase


class Vector2D(VectorBase):

    def __init__(self, x: float, y: float) -> None:
        self.array = np.empty(2)
        self.x = x
        self.y = y

    def rotate(self, angle: float) -> "VectorBase":
        self.array = self.rotated(angle).array
        return self
    
    def rotated(self, angle: float) -> "VectorBase":
        return Vector2D(
            self.x * np.cos(angle) - self.y * np.sin(angle), 
            self.x * np.sin(angle) + self.y * np.cos(angle))
    
    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}
    
    


class Vector3D(VectorBase):

    def __init__(self, x: float, y: float, z: float) -> None:
        self.array = np.empty(3)
        self.x = x
        self.y = y
        self.z = z

    @property
    @abstractmethod
    def z(self) -> float:
        return self.array[2]
    
    @z.setter
    @abstractmethod
    def z(self, value: float) -> None:
        self.array[2] = value

    def rotate(self, angle: float, axis: "VectorBase") -> "VectorBase":
        self = self.rotated(angle, axis)
        return self

    def rotated(self, angle: float, axis: "VectorBase") -> "VectorBase":
        return self.cross(axis) * np.sin(angle) + self * np.cos(angle) + axis * self.dot(axis) * (1 - np.cos(angle))

    def rotate(self, angle: float) -> "VectorBase":
        self.array = self.rotated(angle).array
        return self
    
    def rotated(self, angle: float) -> "VectorBase":
        return Vector2D(
            self.x * np.cos(angle) - self.y * np.sin(angle), 
            self.x * np.sin(angle) + self.y * np.cos(angle))
    
    def angle(self, other: "VectorBase") -> float:
        assert self.magnitude != 0 and other.magnitude != 0
        if self == other: return 0
        return np.arccos(self.dot(other) / (self.magnitude * other.magnitude))
    
    def reflect(self, normal: "VectorBase") -> "VectorBase":
        return self - normal * 2 * self.dot(normal)
    
    def refract(self, normal: "VectorBase", eta: float) -> "VectorBase":
        k = 1.0 - eta * eta * (1.0 - self.dot(normal) * self.dot(normal))
        if k < 0.0:
            return Vector3D.zero()
        else:
            return self * eta - normal * (eta * self.dot(normal) + np.sqrt(k))
    
    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}

    @staticmethod
    def zero() -> "VectorBase":
        return Vector2D._zero(2)
    
    @staticmethod
    def one() -> "VectorBase":
        return Vector2D._one(2)