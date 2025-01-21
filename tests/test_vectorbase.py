import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.geometry.base.vectorbase import VectorBase

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