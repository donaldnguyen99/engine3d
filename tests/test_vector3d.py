import numpy as np
import pytest

from unittest import TestCase

from engine3d.geometry.vector import Vector3D


class TestVector3D(TestCase):
    def test_vector3d___init__(self):
        v = Vector3D(3, 4, 5)
        assert v.x == 3
        assert v.y == 4
        assert v.z == 5
        v.x = 5
        assert v.x == 5
        v.y = 12
        assert v.y == 12
        v.z = 7
        assert v.z == 7
        assert np.all(v.array == np.array([5, 12, 7]))
        assert str(v.__dict__) == "{'array': array([ 5., 12.,  7.]), 'lookup': {'x': 0, 'y': 1, 'z': 2}}"

    def test_vector3d_addition(self):
        v1 = Vector3D(3, 4, 5)
        v2 = Vector3D(5, 6, 7)
        assert v1 + v2 == Vector3D(8, 10, 12)
        v1 += v2
        assert v1 == Vector3D(8, 10, 12)
        v1 += Vector3D(0.5, 0.4, 0.3)
        assert v1 == Vector3D(8.5, 10.4, 12.3)
        
    def test_vector3d_subtraction(self):
        v1 = Vector3D(3, 4, 5)
        v2 = Vector3D(5, 6, 7)
        assert v1 - v2 == Vector3D(-2, -2, -2)
        v1 -= v2
        assert v1 == Vector3D(-2, -2, -2)
        v1 -= Vector3D(0.5, 0.4, 0.3)
        assert v1 == Vector3D(-2.5, -2.4, -2.3)
        
    def test_vector3d_multiplication(self):
        v1 = Vector3D(3, 4, 5)
        v2 = v1 * 3
        assert v2 is not None
        assert v2 == Vector3D(9, 12, 15)
        v3 = 3 * v1
        assert v3 == Vector3D(9, 12, 15)
        v1 *= 2
        assert v1 == Vector3D(6, 8, 10)
        v1 *= 0.5
        assert v1 == Vector3D(3, 4, 5)
        with pytest.raises(NotImplementedError) as e:
            v = Vector3D(1, 2, 3) * Vector3D(3, 4, 5)
        assert e.type == NotImplementedError
        assert "* operation" in str(e.value) and "is not implemented" in str(e.value)
    
    def test_vector3d_division(self):
        v1 = Vector3D(3, 4, 5)
        v2 = v1 / 3
        assert v2 == Vector3D(1, 4/3, 5/3)
        v1 /= 2
        assert v1 == Vector3D(1.5, 2, 2.5)
        v1 /= 0.5
        assert v1 == Vector3D(3, 4, 5)
    
    def test_vector3d_magnitude_squared(self):
        v = Vector3D(3, 4, 5)
        assert v.magnitude_squared == 50
    
    def test_vector3d_magnitude(self):
        v = Vector3D(3, 4, 5)
        assert v.magnitude == np.sqrt(50)
    
    def test_vector3d_magnitude_deleted(self):
        v = Vector3D(3, 4, 5)
        m = v.magnitude
        m2 = v.magnitude_squared
        v.x = 1
        assert "magnitude" not in v.__dict__
        assert "magnitude_squared" not in v.__dict__

    def test_vector3d_cross(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        v = v1.cross(v2)
        assert v == Vector3D(0, 0, 1)

    def test_vector3d_unit___getattr__swizzling(self):
        v = Vector3D(1, 2, 3)
        v.unit = "m"
        assert v.unit == "m"
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3
        assert v.xyy == Vector3D(1, 2, 2)
        assert v.xyx == Vector3D(1, 2, 1)
        assert v.zzx == Vector3D(3, 3, 1)

    def test_vector3d_rotate(self):
        v = Vector3D(1, 2, 3)
        v.rotate(np.pi / 2, Vector3D(0, 0, 1))
        assert v == Vector3D(-2, 1, 3)
        # add more tests
        
    def test_orthonormal_basis(self):
        v = Vector3D(1, 0, 0)
        v1, v2, v3 = v.orthonormal_basis()
        assert v1 == Vector3D(1, 0, 0)
        assert v2 == Vector3D(0, 1, 0)
        assert v3 == Vector3D(0, 0, 1)
        # add more tests

        u = Vector3D(0, 1, 0)
        u1, u2, u3 = u.orthonormal_basis()
        assert u1 == Vector3D(0, 1, 0)
        assert u2 == Vector3D(0, 0, 1)
        assert u3 == Vector3D(1, 0, 0)
        # add more tests
    
    def test_azimuth_elevation_between(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        azimuth, elevation = v1.azimuth_elevation_between(v2)
        assert azimuth == np.pi / 2
        assert elevation == 0

        v3 = Vector3D(0, 0, 1)
        azimuth, elevation = v1.azimuth_elevation_between(v3)
        assert azimuth == 0
        assert elevation == np.pi / 2

        azimuth, elevation = v3.azimuth_elevation_between(v2)
        assert azimuth == 0
        assert elevation == -np.pi / 2
        # add more tests

    def test_rotate_by_azimuth_elevation(self):
        v1 = Vector3D(1, 0, 0)
        v2 = v1.rotate_by_azimuth_elevation(np.pi / 2, 0)
        assert v2 == Vector3D(0, 1, 0)

        v3 = v1.rotate_by_azimuth_elevation(0, np.pi / 2)
        assert v3 == Vector3D(0, 0, 1)

        with pytest.raises(ValueError) as e:
            v4 = v3.rotate_by_azimuth_elevation(0, -np.pi / 2)
        v5 = v3.rotate_by_azimuth_elevation(np.pi / 4, 0)
        assert v5 == Vector3D(0, 0, 1)
        # add more tests