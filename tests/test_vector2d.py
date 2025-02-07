import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.math.vector import Vector2D


class TestVector2D(TestCase):
    def test_vector2d_cached_properties(self):
        v = Vector2D(3, 4)
        assert v.magnitude == 5
        assert v.magnitude_squared == 25
        v.x = 5
        assert v.magnitude == np.sqrt(41)
        assert v.magnitude_squared == 41
        v.y = 12
        assert v.magnitude == 13
        assert v.magnitude_squared == 169
        v.normalize()
        assert np.isclose(v.magnitude, np.sqrt(1.0), atol=Vector2D.EPSILON)
        assert np.isclose(v.magnitude_squared, 1.0, atol=Vector2D.EPSILON)

    def test_vector2d___init__(self):
        v = Vector2D(3, 4)
        assert v.x == 3
        assert v.y == 4
        v.x = 5
        assert v.x == 5
        v.y = 12
        assert v.y == 12
        assert np.all(v.array == np.array([5, 12]))

    def test_vector2d_addition(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1 + v2 == Vector2D(8, 10)
        v1 += v2
        assert v1 == Vector2D(8, 10)
        v1 += Vector2D(0.5, 0.4)
        assert v1 == Vector2D(8.5, 10.4)

    def test_vector2d_subtraction(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1 - v2 == Vector2D(-2, -2)
        v1 -= v2
        assert v1 == Vector2D(-2, -2)
        v1 -= Vector2D(0.5, 0.4)
        assert v1 == Vector2D(-2.5, -2.4)

    def test_vector2d_multiplication(self):
        v1 = Vector2D(3, 4)
        v2 = v1 * 3
        assert v2 == Vector2D(9, 12)
        v3 = 3 * v1
        assert v3 == Vector2D(9, 12)
        v1 *= 2
        assert v1 == Vector2D(6, 8)
        v1 *= 0.5
        assert v1 == Vector2D(3, 4)
        with pytest.raises(NotImplementedError) as e:
            v = Vector2D(1, 2) * Vector2D(3, 4)
            assert e.type == NotImplementedError
            assert "* operation" in str(e.value) and "is not implemented" in str(e.value)

    def test_vector2d_division(self):
        v1 = Vector2D(3, 4)
        v2 = v1 / 2
        assert v2 == Vector2D(1.5, 2)
        v1 /= 2
        assert v1 == Vector2D(1.5, 2)
        v1 /= 0.5
        assert v1 == Vector2D(3, 4)
        with pytest.warns(RuntimeWarning):
            assert v1 / 0 == Vector2D(np.inf, np.inf)
        with pytest.raises(NotImplementedError) as e:
            res = v1 / v2
            print(str(e.value))
            assert e.type == NotImplementedError
            assert "/ operation" in str(e.value) and "is not"

    def test_vector2d_negative(self):
        v = Vector2D(3, 4)
        assert -v == Vector2D(-3, -4)

    def test_vector2d_abs(self):
        v = Vector2D(-3, -4)
        assert abs(v) == 5

    def test_vector2d_eq(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(3.0, 4.0)
        assert v1 == v2
        v3 = Vector2D(3.0000000000000001, 4.0000000000000001)
        assert v1 == v3
        v4 = Vector2D(3.0000000000000001, 4.000000000000004)
        assert v4 == v1
        v5 = Vector2D(3.0000000000000001, 4.000000000000005)
        assert not v1 == v5

    def test_vector2d_not_equal(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(3.0, 4.0)
        assert not v1 != v2
        v3 = Vector2D(3.0000000000000001, 4.0)
        assert not v1 != v3
        v4 = Vector2D(1e10, 0)
        v5 = Vector2D(1e10 + 1e-4, 0)
        assert v4 != v5
        v6 = Vector2D(1e10, 0)
        v7 = Vector2D(1e10 + 1e-5, 0)
        assert not v6 != v7

    def test_vector2d_getitem(self):
        v = Vector2D(3, 4)
        assert v[0] == 3
        assert v[1] == 4
        with pytest.raises(IndexError):
            v[2]

    def test_vector2d___setitem__(self):
        v = Vector2D(3, 4)
        v[0] = 5
        assert v[0] == 5
        v[1] = 12
        assert v[1] == 12
        with pytest.raises(IndexError):
            v[2] = 0

    def test_vector2d___len__(self):
        v = Vector2D(3, 4)
        assert len(v) == 2

    def test_vector2d___iter__(self):
        v = Vector2D(3, 4)
        for i, val in enumerate(v):
            assert val == v[i]

    def test_vector2d___repr__(self):
        v = Vector2D(3, 4)
        assert repr(v) == "Vector2D(3.0, 4.0)"

    def test_vector2d___str__(self):
        v = Vector2D(3, 4)
        assert str(v) == "(3.0, 4.0)"

    def test_vector2d_dot(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1.dot(v2) == 39
        assert v2.dot(v1) == 39
        assert v1.dot(v1) == 25
        assert v2.dot(v2) == 61
        v3 = Vector2D(0, 0.1)
        assert np.isclose(
            v3.dot(v3), 0.01, rtol=Vector2D.EPSILON, atol=Vector2D.EPSILON
        )

    def test_vector2d_cross(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1.cross(v2) == -2
        assert v2.cross(v1) == 2
        assert v1.cross(v1) == 0
        assert v2.cross(v2) == 0


    def test_vector2d_normalize(self):
        v = Vector2D(3, 4)
        v.normalize()
        assert v == Vector2D(0.6, 0.8)
        v = Vector2D(0, 0)
        v1 = v.normalized()
        assert v1 == Vector2D.zero()
        v.normalize()
        assert v == Vector2D.zero()

    def test_vector2d_normalized(self):
        v = Vector2D(3, 4)
        v1 = v.normalized()
        assert v1 == Vector2D(0.6, 0.8)
        v = Vector2D(0, 0)
        v2 = v.normalized()
        assert v2 == Vector2D.zero()
        assert isinstance(v2, Vector2D)

    def test_vector2d_magnitude(self):
        v = Vector2D(3, 4)
        assert v.magnitude == 5
        v.x = 5
        assert v.magnitude == np.sqrt(41)
        v.y = 12
        assert v.magnitude == 13

    def test_vector2d_magnitude_squared(self):
        v = Vector2D(3, 4)
        assert v.magnitude_squared == 25
        v.x = 5
        assert v.magnitude_squared == 41
        v.y = 12
        assert v.magnitude_squared == 169

    def test_vector2d_magnitude_cache_cleared(self):
        v = Vector2D(3, 4)
        v.magnitude
        v.magnitude_squared
        v.x = 5
        assert "magnitude" not in v.__dict__
        assert "magnitude_squared" not in v.__dict__

    def test_vector2d_length(self):
        v = Vector2D(3, 4)
        assert v.length == 5
        v.x = 5
        assert v.length == np.sqrt(41)
        v.y = 12
        assert v.length == 13

    def test_vector2d_distance(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1.distance(v2) == np.sqrt(8)
        assert v2.distance(v1) == np.sqrt(8)
        assert v1.distance(v1) == 0
        assert v2.distance(v2) == 0

    def test_vector2d_distance_squared(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1.distance_squared(v2) == 8
        assert v2.distance_squared(v1) == 8
        assert v1.distance_squared(v1) == 0
        assert v2.distance_squared(v2) == 0

    def test_vector2d_rotate(self):
        v = Vector2D(3, 4)
        v.rotate(np.pi / 2)
        assert v == Vector2D(-4, 3)
        v.rotate(np.pi / 2)
        assert v == Vector2D(-3, -4)
        v.rotate(np.pi / 2)
        assert v == Vector2D(4, -3)
        v.rotate(np.pi / 2)
        assert v == Vector2D(3, 4)
        v.rotate(np.pi)
        assert v == Vector2D(-3, -4)
        v.rotate(np.pi)
        assert v == Vector2D(3, 4)
        v = Vector2D(0, 2)
        v.rotate(np.pi / 4)
        assert v == Vector2D(-np.sqrt(2), np.sqrt(2))
        v.rotate(np.pi / 2)
        assert v == Vector2D(-np.sqrt(2), -np.sqrt(2))
        v.rotate(np.pi / 4)
        assert v == Vector2D(0, -2)
        v.rotate(np.pi / 4)
        assert v == Vector2D(np.sqrt(2), -np.sqrt(2))

    def test_vector2d_rotated(self):
        v = Vector2D(3, 4)
        v1 = v.rotated(np.pi / 2)
        assert v1 == Vector2D(-4, 3)
        v2 = v1.rotated(np.pi / 2)
        assert v2 == Vector2D(-3, -4)
        v3 = v2.rotated(np.pi / 2)
        assert v3 == Vector2D(4, -3)
        v4 = v3.rotated(np.pi / 2)
        assert v4 == Vector2D(3, 4)
        v5 = v4.rotated(np.pi)
        assert v5 == Vector2D(-3, -4)
        v6 = v5.rotated(np.pi)
        assert v6 == Vector2D(3, 4)
        v7 = Vector2D(0, 2)
        v8 = v7.rotated(np.pi / 4)
        assert v8 == Vector2D(-np.sqrt(2), np.sqrt(2))
        v9 = v8.rotated(np.pi / 2)
        assert v9 == Vector2D(-np.sqrt(2), -np.sqrt(2))
        v10 = v9.rotated(np.pi / 4)
        assert v10 == Vector2D(0, -2)
        v11 = v10.rotated(np.pi / 4)
        assert v11 == Vector2D(np.sqrt(2), -np.sqrt(2))

    def test_vector2d_angle(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        assert v1.angle(v2) == np.arccos(39 / (5 * np.sqrt(61)))
        assert v2.angle(v1) == np.arccos(39 / (5 * np.sqrt(61)))
        assert v1.angle(v1) == 0
        assert v2.angle(v2) == 0

    def test_vector2d_lerp(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(5, 6)
        v3 = v1.lerp(v2, 0.5)
        assert v3 == Vector2D(4, 5)
        v4 = v1.lerp(v2, 0)
        assert v4 == v1
        v5 = v1.lerp(v2, 1)
        assert v5 == v2
        v6 = v1.lerp(v2, 2)
        assert v6 == Vector2D(7, 8)

    def test_vector2d_project(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(0, 1)
        v3 = v1.project(v2)
        assert v3 == Vector2D(0, 4)
        v4 = Vector2D(1, 0)
        assert v1.project(v4) == Vector2D(3, 0)
        v5 = Vector2D(-0.5, 0.3)
        assert v5.project(v4) == Vector2D(-0.5, 0)

    def test_vector2d_reject(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(0, 1)
        v3 = v1.reject(v2)
        assert v3 == Vector2D(3, 0)
        v4 = Vector2D(1, 0)
        assert v1.reject(v4) == Vector2D(0, 4)
        v5 = Vector2D(-0.5, 0.3)
        assert v5.reject(v4) == Vector2D(0, 0.3)

    def test_vector2d_reflect(self):
        v1 = Vector2D(3, 4)
        v2 = Vector2D(0, -1)
        v3 = v1.reflect(v2)
        assert v3 == Vector2D(3, -4)
        v4 = Vector2D(-1, 0)
        assert v1.reflect(v4) == Vector2D(-3, 4)
        v5 = Vector2D(-0.5, 0.3)
        assert v5.reflect(v4) == Vector2D(0.5, 0.3)

    def test_vector2d_refract(self):
        v1 = Vector2D(3, -4)
        v2 = Vector2D(np.sin(np.pi / 4), -np.cos(np.pi / 4)) * 3
        n = Vector2D(0, 1)
        assert v1.refract(n, 1) == Vector2D(3, -4)
        v3 = v1.refract(n, 0)
        assert isinstance(v3, Vector2D)
        assert v3 == Vector2D.zero()

        v4 = v2.refract(n, 0.67)
        assert isinstance(v4, Vector2D)
        assert v4 == Vector2D(1.42128463018496065, -2.64195950006808422)

        negn = Vector2D(0, -1)
        v5 = v2.reflect(Vector2D(0, 1))
        assert isinstance(v5, Vector2D)
        assert v5 == Vector2D(np.sin(np.pi / 4), np.cos(np.pi / 4)) * 3
        v6 = v5.refract(negn, 0.67)
        assert isinstance(v6, Vector2D)
        assert v6 == Vector2D(1.42128463018496065, 2.64195950006808422)
        v7 = v4.reflect(Vector2D(0, 1)).refract(negn, 1 / 0.67)
        assert isinstance(v7, Vector2D)
        assert v7 == Vector2D(np.sin(np.pi / 4), np.cos(np.pi / 4)) * 3

    def test_vector2d_refract2(self):
        # edge cases cosi >= 0 and k < 0
        v = Vector2D(10, -1)
        n = Vector2D(0, -1)
        eta = 1.5
        v2 = v.refract(n, eta)
        assert v2 == Vector2D.zero()

    def test_vector2d_to_tuple(self):
        v = Vector2D(3, 4)
        assert v.to_tuple() == (3, 4)

    def test_vector2d_to_list(self):
        v = Vector2D(3, 4)
        assert v.to_list() == [3, 4]

    def test_vector2d_to_set(self):
        v = Vector2D(3, 4)
        assert v.to_set() == {3, 4}

    def test_vector2d_to_dict(self):
        v = Vector2D(3, 4)
        assert v.to_dict() == {"x": 3, "y": 4}

    def test_vector2d_to_bytes(self):
        v = Vector2D(3, 4)
        assert (
            v.to_bytes()
            == b"\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@"
        )

    def test_vector2d_to_array(self):
        v = Vector2D(3, 4)
        assert np.array_equal(v.to_array(), np.array([3, 4]))
        arr = v.to_array()
        v.x = 5
        assert not np.array_equal(v.to_array(), arr)

    def test_vector2d_copy(self):
        v = Vector2D(3, 4)
        v1 = v.copy()
        assert v1 == v
        v.x = 5
        assert v1 != v
        v1.x = 5
        assert v1 == v

    def test_vector2d_zero(self):
        v = Vector2D(3, 4)
        assert v != Vector2D.zero()
        v1 = Vector2D(0, 0)
        assert v1 == Vector2D.zero()

    def test_vector2d_one(self):
        v = Vector2D(3, 4)
        assert v != Vector2D.one()
        v1 = Vector2D(1, 1)
        assert v1 == Vector2D.one()

    def test_vector2d_unit(self):
        v = Vector2D(3, 4)
        assert v != v.unit()
        v1 = Vector2D(0.6, 0.8)
        assert v1 == v.unit()

    def test_vector2d_unit___setattr__(self):
        v = Vector2D(3, 4)
        v.unit = "cm"
        assert hasattr(v, "unit")
        assert v.unit == "cm"
    
    def test_vector2d_unit___getattr__swizzling(self):
        v = Vector2D(3, 4)
        v.unit = "cm"
        assert v.unit == "cm"
        assert getattr(v, "unit") == v.__dict__["unit"]
        assert v.__getattr__("unit") == v.__dict__["unit"]
        assert v.xx == Vector2D(3, 3)
        assert v.yy == Vector2D(4, 4)
        assert v.xy == Vector2D(3, 4)
        assert v.yx == Vector2D(4, 3)
        assert v.x == 3
        with pytest.raises(AttributeError) as e:
            a = v.z
        assert "'Vector2D' object has no attribute" in str(e.value)
        with pytest.raises(AttributeError) as e2:
            a = v.xz
        assert "'Vector2D' object has no attribute" in str(e2.value)
