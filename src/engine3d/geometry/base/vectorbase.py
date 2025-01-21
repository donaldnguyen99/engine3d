from abc import ABC, abstractmethod
from collections import abc
from functools import cached_property
from typing import Union

import inspect
import numpy as np


class VectorBase(ABC):

    dim: int
    EPSILON: float = 1e-15

    def __init__(self, *args: float) -> None:
        """
        Initialize a vector.
        """
        self.array = np.empty(self.dim)
        arg_names_and_vals = [(key, value) for key, value in inspect.getcallargs(self.__init__, *args).items() if key != "self"]
        self.lookup = {arg[0]: i for i, arg in enumerate(arg_names_and_vals)}
        for i, arg in enumerate(arg_names_and_vals):
            setattr(self, arg[0], arg[1])
        print("print from vectorbase: ", str(arg_names_and_vals))

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns the dimension of this vector.

        Returns:
            int: The dimension of this vector.
        """
        pass
    
    def __setitem__(self, index: int, value: float) -> None:
        """
        Set the value of the vector at the given index.

        Args:
            index (int): The index of the value to set.
            value (float): The value to set.
        """
        self.array[index] = value
        if hasattr(self, "magnitude"):
            del self.magnitude
        if hasattr(self, "magnitude_squared"):
            del self.magnitude_squared

    def __getitem__(self, index: int) -> float:
        """
        Get the value of the vector at the given index.

        Args:
            index (int): The index of the value to get.

        Returns:
            float: The value of the vector at the given index.
        """
        return self.array[index]

    def __setattr__(self, name, value):
        if name in {"array", "lookup"}:
            super().__setattr__(name, value)
            return
        
        # The above code is needed to avoid infinite recursion
        if name in self.lookup:
            self.array[self.lookup[name]] = value
            if hasattr(self, "magnitude"):
                del self.magnitude
            if hasattr(self, "magnitude_squared"):
                del self.magnitude_squared
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in {"array", "lookup"}:
            return super().__getattr__(name)
        
        if name in self.lookup:
            return self.array[self.lookup[name]]
        else:
            return super().__getattr__(name)

    def __add__(self, other: "VectorBase") -> "VectorBase":
        """
        Returns the vector sum of this vector and another vector.

        Args:
            other (VectorBase): The vector to add to this vector.

        Returns:
            VectorBase: The vector sum of this vector and the other vector.
        """
        return self.__class__(*(self.array + other.array))

    def __sub__(self, other: "VectorBase") -> "VectorBase":
        """
        Subtract another vector from this vector.

        Args:
            other (VectorBase): The vector to subtract from this vector.

        Returns:
            VectorBase: A new vector representing the result of the subtraction.
        """
        return self.__class__(*(self.array - other.array))

    def __mul__(self, other) -> "VectorBase":
        """
        Multiply this vector by a scalar, vector, or matrix.

        Args:
            other (float): The scalar to multiply this vector by.

        Returns:
            VectorBase: A new vector representing the result of the multiplication.
        """
        try:
            if isinstance(other, (float, int)):
                return self.__class__(*(self.array * other))
        except TypeError:
            return NotImplemented

    def __rmul__(self, other) -> "VectorBase":
        """
        Multiply this vector by a scalar.

        Args:
            other (float): The scalar to multiply this vector by.

        Returns:
            VectorBase: A new vector representing the result of the multiplication.
        """
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "VectorBase":
        """
        Divide this vector by a scalar.

        Args:
            other (float): The scalar to divide this vector by.

        Returns:
            VectorBase: A new vector representing the result of the division.
        """
        try:
            if isinstance(other, (float, int)):
                return self.__class__(*(self.array / other))
        except TypeError:
            return NotImplemented

    def __neg__(self) -> "VectorBase":
        """
        Negate this vector.

        Returns:
            VectorBase: A new vector representing the negation of this vector.
        """
        return self.__class__(*(-self.array))

    def __abs__(self) -> float:
        """
        Returns the magnitude of this vector.

        Returns:
            float: The magnitude of this vector.
        """
        return self.magnitude

    def __eq__(self, other: "VectorBase") -> bool:
        """
        Check if this vector is equal to another vector.

        Args:
            other (VectorBase): The vector to check for equality.

        Returns:
            bool: True if the vectors are equal, False otherwise.
        """
        return np.allclose(
            self.array, other.array, rtol=self.EPSILON, atol=self.EPSILON
        )

    def __ne__(self, other: "VectorBase") -> bool:
        """
        Check if this vector is not equal to another vector.

        Args:
            other (VectorBase): The vector to check for inequality.

        Returns:
            bool: True if the vectors are not equal, False otherwise.
        """
        return not self == other

    def __getitem__(self, index: int) -> float:
        """
        Get the value of the vector at the given index.

        Args:
            index (int): The index of the value to get.

        Returns:
            float: The value of the vector at the given index.
        """
        return self.array[index]

    def __setitem__(self, index: int, value: float) -> None:
        """
        Set the value of the vector at the given index.

        Args:
            index (int): The index of the value to set.
            value (float): The value to set.
        """
        self.array[index] = value
        if hasattr(self, "magnitude"):
            del self.magnitude

    def __len__(self) -> int:
        """
        Returns the number of components in this vector.

        Returns:
            int: The number of components in this vector.
        """
        return len(self.array)

    def __iter__(self) -> abc.Iterator[float]:
        """
        Returns an iterator over the components of this vector.

        Returns:
            Iterator[float]: An iterator over the components of this vector.
        """
        return iter(self.array)

    def __repr__(self) -> str:
        """
        Returns a string representation of this vector.

        Returns:
            str: A string representation of this vector.
        """
        return f"{self.__class__.__name__}{str(self)}"

    def __str__(self) -> str:
        """
        Returns a string representation of this vector.

        Returns:
            str: A string representation of this vector.
        """
        return f"({', '.join(str(x) for x in self.array)})"

    def dot(self, other: "VectorBase") -> float:
        """
        Returns the dot product of this vector and another vector.

        Args:
            other (VectorBase): The vector to dot with this vector.

        Returns:
            float: The dot product of this vector and the other vector.
        """
        return np.dot(self.array, other.array)

    def cross(self, other: "VectorBase") -> Union[float, "VectorBase"]:
        """
        Returns the cross product of this vector and another vector.

        Args:
            other (VectorBase): The vector to cross with this vector.

        Returns:
            float: The cross product of this vector and the other vector.
        """
        if self.dim != other.dim:
            raise ValueError("Cross product is only defined for 2D or 3D vectors of the same dimension.")
        if self.dim == 2:
            return np.cross(self.array, other.array)
        elif self.dim == 3:
            return self.__class__(*(np.cross(self.array, other.array)))
        else:
            raise ValueError("Cross product is only defined for 2D and 3D vectors.")

    def normalize(self) -> "VectorBase":
        """
        Normalizes this vector in place and returns it.

        Returns:
            None
        """
        if self.magnitude_squared == 0:
            return self
        self.array /= self.magnitude
        del self.magnitude
        del self.magnitude_squared
        return self

    def normalized(self) -> "VectorBase":
        """
        Returns a normalized version of this vector.

        Returns:
            VectorBase: A normalized version of this vector.
        """
        if self.magnitude_squared == 0:
            return self.zero()
        return self.__class__(*(self.array / self.magnitude))

    @cached_property
    def magnitude(self) -> float:
        """
        Returns the magnitude of this vector.

        Returns:
            float: The magnitude of this vector.
        """
        return np.sqrt(self.magnitude_squared)

    @cached_property
    def magnitude_squared(self) -> float:
        """
        Returns the squared magnitude of this vector.

        Returns:
            float: The squared magnitude of this vector.
        """
        return self.array.dot(self.array)

    @property
    def length(self) -> float:
        """
        Returns the length of this vector.

        Returns:
            float: The length of this vector.
        """
        return self.magnitude

    def distance(self, other: "VectorBase") -> float:
        """
        Returns the distance between this vector and another vector.

        Args:
            other (VectorBase): The vector to calculate the distance to.

        Returns:
            float: The distance between this vector and the other vector.
        """
        return np.sqrt(self.distance_squared(other))

    def distance_squared(self, other: "VectorBase") -> float:
        """
        Returns the squared distance between this vector and another vector.

        Args:
            other (VectorBase): The vector to calculate the distance to.

        Returns:
            float: The squared distance between this vector and the other vector.
        """
        return (self.array - other.array).dot(self.array - other.array)

    @abstractmethod
    def rotate(self, *args) -> "VectorBase":
        """
        Rotates the vector around the given axis in place and returns it.

        Returns:
            The vector rotated around the given axis.
        """
        pass

    @abstractmethod
    def rotated(self, *args) -> "VectorBase":
        """
        Returns a new vector rotated around the given axis.

        Returns:
            VectorBase: A new vector rotated around the given axis.
        """
        pass

    def angle(self, other: "VectorBase") -> float:
        """
        Returns the angle between this vector and another vector.

        Args:
            other (VectorBase): The vector to calculate the angle to.

        Returns:
            float: The angle between this vector and the other vector.
        """
        assert self.magnitude != 0 and other.magnitude != 0
        if self == other:
            return 0
        return np.arccos(self.dot(other) / (self.magnitude * other.magnitude))

    def lerp(self, other: "VectorBase", t: float) -> "VectorBase":
        """
        Returns a new vector linearly interpolated between this vector and another vector.

        Args:
            other (VectorBase): The vector to interpolate to.
            t (float): The interpolation factor.

        Returns:
            VectorBase: A new vector linearly interpolated between this vector and the other vector.
        """
        return self + (other - self) * t

    def project(self, other: "VectorBase") -> "VectorBase":
        """
        Returns a new vector projected onto another vector.

        Args:
            other (VectorBase): The vector to project onto.

        Returns:
            VectorBase: A new vector projected onto the other vector.
        """
        return other * (self.dot(other) / other.dot(other))

    def reject(self, other: "VectorBase") -> "VectorBase":
        """
        Returns a new vector rejected from another vector.

        Args:
            other (VectorBase): The vector to reject from.

        Returns:
            VectorBase: A new vector rejected from the other vector.
        """
        return self - self.project(other)

    def reflect(self, normal: "VectorBase") -> "VectorBase":
        """
        Returns a new vector reflected across a normal vector.

        Args:
            normal (VectorBase): The normal vector to reflect across.

        Returns:
            VectorBase: A new vector reflected across the normal vector.
        """
        normal_normalized = normal.normalized()
        return self - normal_normalized * 2 * self.dot(normal_normalized)

    def refract(self, normal: "VectorBase", eta: float) -> "VectorBase":
        """
        Returns a new vector refracted across a normal vector.

        Args:
            normal (VectorBase): The normal vector to refract across.
            eta (float): The relative index of refraction between media of the
            incident and refracted rays, respectively.

        Returns:
            VectorBase: A new vector refracted across the normal vector.
        """
        # Self and normal are not normalized
        if eta == 0.0:
            return self.__class__.zero()
        self_normalized = self.normalized()
        normal_normalized = normal.normalized()
        cosi = normal_normalized.dot(self_normalized)
        if cosi < 0.0:
            cosi = -cosi
        else:
            normal_normalized = -normal_normalized
        k = 1.0 - eta * eta * (1.0 - cosi * cosi)
        if k < 0.0:
            return self.__class__.zero()
        else:
            return (
                self_normalized * eta + normal_normalized * (eta * cosi - k**0.5)
            ) * self.magnitude

    def to_tuple(self) -> tuple[float, float]:
        """
        Returns a tuple representation of this vector.

        Returns:
            tuple[float, float]: A tuple representation of this vector.
        """
        return tuple(self.array)

    def to_list(self) -> list[float]:
        """
        Returns a list representation of this vector.

        Returns:
            list[float]: A list representation of this vector.
        """
        return list(self.array)

    def to_set(self) -> set[float]:
        """
        Returns a set representation of this vector.

        Returns:
            set[float]: A set representation of this vector.
        """
        return set(self.array)

    def to_dict(self) -> dict[str, float]:
        """
        Returns a dictionary representation of this vector.

        Returns:
            dict[str, float]: A dictionary representation of this vector.
        """
        return {f"{i}": self.array[self.lookup[i]] for i in self.lookup}

    def to_bytes(self) -> bytes:
        """
        Returns a bytes representation of this vector.

        Returns:
            bytes: A bytes representation of this vector.
        """
        return self.array.tobytes()

    def to_array(self) -> np.ndarray:
        """
        Returns an array representation of this vector.

        Returns:
            VectorBase: An array representation of this vector.
        """
        return self.array.copy() 

    def copy(self) -> "VectorBase":
        """
        Returns a copy of this vector.

        Returns:
            VectorBase: A copy of this vector.
        """
        return self.__class__(*self.array)

    @classmethod
    def zero(cls) -> "VectorBase":
        """
        Returns a zero vector.

        Returns:
            VectorBase: A zero vector.
        """
        return cls(*np.zeros(cls.dim))

    @classmethod
    def one(cls) -> "VectorBase":
        """
        Returns a one vector.

        Returns:
            VectorBase: A one vector.
        """
        return cls(*np.ones(cls.dim))

    def unit(self) -> "VectorBase":
        """
        Returns a unit vector.

        Returns:
            VectorBase: A unit vector.
        """
        return self.__class__(*(self.array / self.magnitude))
