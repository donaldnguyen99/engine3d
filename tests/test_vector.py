import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.math.base.vectorbase import VectorBase
from engine3d.math.vector import Vector2D, Vector3D

class TestVector(TestCase):
    def test_vector2d_add_vector2d_vector3d(self):
        v1 = Vector2D(1, 2)
        v2 = Vector3D(3, 4, 5)
        with pytest.raises(ValueError) as e:
            v3 = v1 + v2
        assert e.type == ValueError
        assert "operands could not be broadcast together with shapes (2,) (3,)" in str(e.value)
    
    def test_vector2d_add_vector3d_2(self):
        v1 = Vector3D(3, 4, 5)
        v2 = Vector2D(1, 2)
        with pytest.raises(ValueError) as e:
            v3 = v1 + v2
        assert e.type == ValueError
        assert "operands could not be broadcast together with shapes (3,) (2,)" in str(e.value)
    
    def test_vector2d_mul_vector3d(self):
        v1 = Vector2D(2, 3)
        v2 = Vector3D(1, 2, 3)
        with pytest.raises(NotImplementedError) as e:
            v = v1 * v2
        assert "* operation between" in str(e.value) and "is not implemented" in str(e.value)
    
    def test_vector2d_truediv_vector3d(self):
        v1 = Vector2D(2, 3)
        v2 = Vector3D(1, 2, 3)
        with pytest.raises(NotImplementedError) as e:
            v = v1 / v2
        assert "/ operation between" in str(e.value) and "is not implemented" in str(e.value)

    def test_vector2d_cross_vector3d(self):
        v1 = Vector2D(2, 3)
        v2 = Vector3D(1, 2, 3)
        with pytest.raises(ValueError) as e:
            v = v1.cross(v2)
        assert e.type == ValueError
        assert "Cross product is only defined for 2D or 3D vectors" in str(e.value)
    
    def test_vector4d_cross_vector4d(self):
        class Vector5D(VectorBase):
            dim = 5
            def __init__(self, x, y, z, w, v):
                super().__init__(x, y, z, w, v)
            def __mul__(self, other):
                pass
            def rotate(self, *args):
                pass
            def rotated(self, *args):
                pass
        v1 = Vector5D(1, 2, 3, 4, 5)
        v2 = Vector5D(0, 1, 2, 3, 4)
        with pytest.raises(ValueError) as e:
            v1.cross(v2)
        assert e.type == ValueError
        assert "Cross product is only defined for 2D and 3D vectors" in str(e.value)
    
    def test_vector3d_swizzle_vector2d(self):
        v = Vector3D(1, 2, 3)
        assert v.xy == Vector2D(1, 2)
        assert v.yz == Vector2D(2, 3)
        assert v.xz == Vector2D(1, 3)
        with pytest.raises(AttributeError) as e:
            v.xxyy
        assert e.type == AttributeError
        assert "has no attribute" in str(e.value) and "xxyy" in str(e.value) 

    
    def test_vectorbase_find_subclasses_with_value(self):
        class VectorBase2(VectorBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.dim = 2

            def rotate(self, angle, axis):
                pass

            def rotated(self, angle, axis):
                pass
        assert set([Vector2D]) <= set(VectorBase.find_subclasses_with_value("dim", 2))
        assert set([Vector3D]) <= set(VectorBase.find_subclasses_with_value("dim", 3))