import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.math.base.vectorbase import VectorBase
from engine3d.math.base.matrixbase import MatrixBase
from engine3d.math.matrix import *

class TestMatrix(TestCase):
    def test_matrix_mult_vector(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        class MyVector(VectorBase):
            dim = 2
            def __init__(self, a, b):
                super().__init__(a, b)
            def rotate(self, angle, axis):
                pass
            def rotated(self, *args):
                pass
            def __mul__(self, other):
                return super().__mul__(other)
        m = MyMatrix([[1, 2], [3, 4]])
        v = MyVector(2, 3)
        assert m @ v == MyVector(8, 18)
    
    