import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.geometry.base.vectorbase import VectorBase
from engine3d.geometry.vector import Vector2D, Vector3D

class TestVector(TestCase):
    def test_vector2d_add_vector2d_vector3d(self):
        v1 = Vector2D(1, 2)
        v2 = Vector3D(3, 4, 5)
        with pytest.raises(ValueError) as e:
            v3 = v1 + v2
        assert e.type == ValueError
        assert "operands could not be broadcast together with shapes (2,) (3,)" in str(e.value)
    
    def test_vector2d_add_vector2d_vector3d_2(self):
        v1 = Vector3D(3, 4, 5)
        v2 = Vector2D(1, 2)
        with pytest.raises(ValueError) as e:
            v3 = v1 + v2
        assert e.type == ValueError
        assert "operands could not be broadcast together with shapes (3,) (2,)" in str(e.value)
    
    def test_vector2d___getattr__super(self):
        v = Vector2D(3, 4)
        assert np.all(getattr(v, "array") == np.array([3, 4]))
        assert np.all(v.array == np.array([3, 4]))

    def test_vectorbase_dim(self):
        with pytest.raises(Exception) as e:
            VectorBase.dim()
        assert e.type == TypeError
        assert "property" in str(e.value) and "not callable" in str(e.value)