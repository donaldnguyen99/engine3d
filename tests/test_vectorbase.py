import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.math.base.vectorbase import VectorBase

class TestVectorBase(TestCase):
    def test_vectorbase___init__(self):
        with pytest.raises(TypeError) as e:
            v = VectorBase(3, 4, 5)
        assert e.type == TypeError
        assert "Can't instantiate abstract class VectorBase" in str(e.value)
        for method in ["dim", "rotate", "rotated"]:
            assert method in str(e.value)

    def test_vectorbase___init__2(self):
        class VectorBase2(VectorBase):
            dim = 3

            def rotate(self, angle, axis):
                pass

        with pytest.raises(TypeError) as e:
            v = VectorBase2(3, 4, 5)
        assert e.type == TypeError
        assert "Can't instantiate abstract class" in str(e.value)
        assert "rotated" in str(e.value)
    
    def test_vectorbase___init__3(self):
        class VectorBase2(VectorBase):
            def __init__(self, *args):
                super().__init__(*args)
                self.dim = 2

            def rotate(self, angle, axis):
                pass

            def rotated(self, angle, axis):
                pass

        with pytest.raises(TypeError) as e:
            v = VectorBase2(3, 4, 5)
        assert e.type == TypeError
        assert "Can't instantiate abstract class VectorBase2" in str(e.value)
        assert "dim" in str(e.value)
        assert "__mul__" in str(e.value)

    def test_vectorbase_dim(self):
        with pytest.raises(Exception) as e:
            VectorBase.dim()
        assert e.type == TypeError
        assert "property" in str(e.value) and "not callable" in str(e.value)

    def test_vectorbase_abstract_method_pass(self):
        class VectorBase2(VectorBase):
            dim = 2
            def __init__(self, a, b):
                super().__init__(a, b)
            
            def get_dim(self):
                return super().dim
            
            def __mul__(self, other):
                return super().__mul__(other)

            def rotate(self, angle, axis):
                return super().rotate(angle, axis)

            def rotated(self, angle, axis):
                return super().rotated(angle, axis)
        assert VectorBase2(1, 2).get_dim() == None
        assert VectorBase2(1, 2).rotate(0, 0) == None
        assert VectorBase2(1, 2).rotated(0, 0) == None

