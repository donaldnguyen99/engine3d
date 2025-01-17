from abc import ABC, abstractmethod
from collections import abc
from functools import cached_property

import numpy as np

from engine3d.geometry.base.vectorbase import VectorBase

class MatrixBase(ABC):

    array: np.ndarray

    def __init__(self, *args) -> None:
        """
        Initialize the matrix.

        Args:
            array (np.ndarray): The array to initialize the matrix with.
        """
        if len(args) == 0:
            self.array = np.zeros((2, 2))
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.array = args[0]
            elif isinstance(args[0], MatrixBase):
                self.array = args[0].array
            elif isinstance(args[0], tuple) and len(args[0]) == 2 and \
                 isinstance(args[0][0], int) and \
                 isinstance(args[0][1], int):
                self.array = np.zeros(args[0])
            elif isinstance(args[0], list):
                self.array = np.array(args[0])
            else:
                raise TypeError(f"Cannot create a matrix from {args[0]}")
        elif len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], int):
                self.array = np.zeros(args)
            else:
                raise TypeError(f"Cannot create a matrix from {args}")
        else:
            raise TypeError(f"Cannot create a matrix from {args}")

    @property
    @abstractmethod
    def dim(self) -> tuple[int, ...]:
        """
        Get the dimensions of the matrix.

        Returns:
            tuple[int, ...]: The dimensions of the matrix.
        """
        pass

    def __getitem__(self, key: tuple[int, ...]) -> float:
        """
        Get the value at the given index.

        Args:
            key (tuple[int, int]): The index of the value to get.

        Returns:
            float: The value at the given index.
        """
        return self.array[key]
    
    def __setitem__(self, key: tuple[int, ...], value: float) -> None:
        """
        Set the value at the given index.

        Args:
            key (tuple[int, int]): The index of the value to set.
            value (float): The value to set at the given index.
        """
        self.array[key] = value
        if hasattr(self, "determinant"):
            del self.determinant
        if hasattr(self, "trace"):
            del self.trace

    def __repr__(self) -> str:
        """
        Returns a string representation of the matrix.

        Returns:
            str: A string representation of the matrix.
        """
        return f"{self.__class__.__name__}({self.__str__()})"
    
    def __str__(self) -> str:
        """
        Returns a string representation of the matrix.

        Returns:
            str: A string representation of the matrix.
        """
        return self.array.__str__()
    
    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the matrix. This is a tuple of the number of rows and columns.

        Returns:
            tuple[int, int]: The shape of the matrix as a tuple of the number of rows and columns.
        """
        return self.array.shape
    
    @property
    def T(self) -> "MatrixBase":
        """
        Returns the transpose of the matrix.

        Returns:
            MatrixBase: The transpose of the matrix.
        """
        return self.__class__(self.array.T)
    
    def __matmul__(self, other: "MatrixBase" | "VectorBase") -> "MatrixBase":
        """
        Returns the matrix product of this matrix and another matrix.

        Args:
            other (MatrixBase): The other matrix to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the other matrix.
        """
        try:
            if isinstance(other, MatrixBase):
                return self.__class__(self.array @ other.array)
            elif isinstance(other, VectorBase):
                return other.__class__(self.array @ other.array)
        except:
            raise TypeError(f"Cannot multiply {self.__class__.__name__} by {other.__class__.__name__}.")
    
    def __rmatmul__(self, other: "MatrixBase" | "VectorBase") -> "MatrixBase":
        """
        Returns the matrix product of this matrix and another matrix.

        Args:
            other (MatrixBase): The other matrix to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the other matrix.
        """
        try:
            if isinstance(other, MatrixBase):
                return self.__class__(other.array @ self.array)
            elif isinstance(other, VectorBase):
                return other.__class__(other.array @ self.array)
        except:
            raise TypeError(f"Cannot multiply {other.__class__.__name__} by {self.__class__.__name__}.")

    def __imatmul__(self, other: "MatrixBase" | "VectorBase") -> "MatrixBase":
        """
        Returns the matrix product of this matrix and another matrix.

        Args:
            other (MatrixBase): The other matrix to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the other matrix.
        """
        self.array = self.array @ other.array
        return self
    
    def __mul__(self, other: float | int) -> "MatrixBase":
        """
        Returns the matrix product of this matrix and a scalar.

        Args:
            other (float | int): The scalar to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the scalar.
        """
        return self.__class__(self.array * other)
    
    def __rmul__(self, other: float | int) -> "MatrixBase":
        """
        Returns the matrix product of this matrix and a scalar.

        Args:
            other (float | int): The scalar to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the scalar.
        """
        return self.__mul__(other)
    
    def __imul__(self, other: float | int) -> "MatrixBase":
        """
        Returns the matrix product of this matrix and a scalar.

        Args:
            other (float | int): The scalar to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the scalar.
        """
        self.array = self.array * other
        return self
    
    def __truediv__(self, other: float | int) -> "MatrixBase":
        """
        Returns the matrix product of this matrix and a scalar.

        Args:
            other (float | int): The scalar to multiply this matrix by.

        Returns:
            MatrixBase: The matrix product of this matrix and the scalar.
        """
        return self.__class__(self.array / other)
    
    def __neg__(self) -> "MatrixBase":
        """
        Returns the negative of the matrix.

        Returns:
            MatrixBase: The negative of the matrix.
        """
        return self.__class__(-self.array)

    def __abs__(self) -> "MatrixBase":
        """
        Returns the absolute value of the matrix (the determinant).

        Returns:
            MatrixBase: The absolute value of the matrix.
        """
        return self.determinant
    
    def __add__(self, other: "MatrixBase") -> "MatrixBase":
        """
        Returns the sum of this matrix and another matrix.

        Args:
            other (MatrixBase): The other matrix to add to this matrix.

        Returns:
            MatrixBase: The sum of this matrix and the other matrix.
        """
        return self.__class__(self.array + other.array)
    
    def __sub__(self, other: "MatrixBase") -> "MatrixBase":
        """
        Returns the difference of this matrix and another matrix.

        Args:
            other (MatrixBase): The other matrix to subtract from this matrix.

        Returns:
            MatrixBase: The difference of this matrix and the other matrix.
        """
        return self.__class__(self.array - other.array)
    
    @cached_property
    def determinant(self) -> float:
        """
        Returns the determinant of the matrix.

        Returns:
            float: The determinant of the matrix.
        """
        return np.linalg.det(self.array)
    
    @property
    def transpose(self) -> "MatrixBase":
        """
        Returns the transpose of the matrix.

        Returns:
            MatrixBase: The transpose of the matrix.
        """
        return self.T
    
    @cached_property
    def trace(self) -> float:
        """
        Returns the trace of the matrix.

        Returns:
            float: The trace of the matrix.
        """
        return np.trace(self.array)
    
    def get_row(self, row: int) -> "VectorBase":
        """
        Returns the row at the given index.

        Args:
            row (int): The index of the row to get.

        Returns:
            VectorBase: The row at the given index.
        """
        return self.array[row]
    
    def get_column(self, column: int) -> "VectorBase":
        """
        Returns the column at the given index.

        Args:
            column (int): The index of the column to get.

        Returns:
            VectorBase: The column at the given index.
        """
        return self.array[:, column]
