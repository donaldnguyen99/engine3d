import numpy as np
import pytest

# For vscode users to detect both test classes
from unittest import TestCase

from engine3d.geometry.base.matrixbase import MatrixBase

class TestMatrixBase(TestCase):
    def test_matrixbase___init__(self):
        class MyMatrix(MatrixBase):
            
            dim = (2, 2)

            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m = MyMatrix()
        assert m.array.shape == (2, 2)
        assert np.all(m.array == np.zeros((2, 2)))
        assert m.dim == (2, 2)

        with pytest.raises(ValueError) as e:
            m1 = MyMatrix(2)
        assert str(e.value) == "Cannot create a matrix from 2"

        m1 = MyMatrix(np.ones((2, 2)))
        assert m1.array.shape == (2, 2)
        assert np.all(m1.array == np.ones((2, 2)))

        m2 = MyMatrix(MyMatrix())
        assert m2.array.shape == (2, 2)
        assert np.all(m2.array == np.zeros((2, 2)))

        m3 = MyMatrix((2, 2))
        assert m3.array.shape == (2, 2)
        assert np.all(m3.array == np.zeros((2, 2)))

        m4 = MyMatrix([[1, 2], [3, 4]])
        assert m4.array.shape == (2, 2)
        assert np.all(m4.array == np.array([[1, 2], [3, 4]]))

        m5 = MyMatrix(2, 2)
        assert m5.array.shape == (2, 2)
        assert np.all(m5.array == np.zeros((2, 2)))

        with pytest.raises(ValueError) as e:
            m6 = MyMatrix(2.1, 2.2)
            assert str(e.value) == "Cannot create a matrix from (2.1, 2.2)"

        with pytest.raises(ValueError) as e:
            m7 = MyMatrix(2, 2, 2)
        assert str(e.value) == "Cannot create a matrix from (2, 2, 2)"

    def test_matrixbase_dim(self):
        with pytest.raises(Exception) as e:
            MatrixBase.dim()
        assert e.type == TypeError
        assert "property" in str(e.value) and "not callable" in str(e.value)

        class MyMatrix(MatrixBase):
            dim = 2
            def __init__(self, *args) -> None:
                super().__init__(*args)

        with pytest.raises(AssertionError) as e:
            m = MyMatrix()
        assert str(e.value) == "dim must be a tuple"

        class MyMatrix(MatrixBase):
            dim = (2, 2, 3)
            def __init__(self, *args) -> None:
                super().__init__(*args)

        with pytest.raises(AssertionError) as e:
            m = MyMatrix()
        assert str(e.value) == "dim must be a tuple of length 2"

        class MyMatrix(MatrixBase):
            dim = (2.6, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        with pytest.raises(AssertionError) as e:
            m = MyMatrix()
        assert str(e.value) == "dim must be a tuple of integers"

    def test_matrixbase_methods_with_pass(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
            def get_dim(self):
                return super().dim
        
        m = MyMatrix()
        assert m.get_dim() == None
    
    def test_matrixbase_set_get(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m = MyMatrix([[1, 2], [3, 4]])
        assert m[0, 0] == 1
        assert m[0, 1] == 2
        assert m[1, 0] == 3
        assert m[1, 1] == 4

        old_trace = m.trace
        old_det = m.determinant
        m[0, 0] = 5
        assert m[0, 0] == 5
        assert m == MyMatrix([[5, 2], [3, 4]])
        assert "trace" not in m.__dict__
        assert "determinant" not in m.__dict__

        

    def test_matrixbase_repr_str(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m = MyMatrix([[1, 2], [3, 4]])
        assert repr(m) == f"MyMatrix({str(np.array([[1, 2], [3, 4]]))})"
        assert str(m) == str(np.array([[1, 2], [3, 4]]))

    def test_matrixbase_shape(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m = MyMatrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2)

    def test_matrixbase_T(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m = MyMatrix([[1, 2], [3, 4]])
        assert np.all(m.T.array == np.array([[1, 3], [2, 4]]))
        assert np.all(m.transpose.array == np.array([[1, 3], [2, 4]]))

    def test_matrixbase___matmul__(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m1 = MyMatrix([[1, 2], [3, 4]])
        m2 = MyMatrix([[1, 2], [3, 4]])
        m3 = m1 @ m2
        assert m3.array.shape == (2, 2)
        assert np.all(m3.array == np.array([[7, 10], [15, 22]]))
        m3 @= m2
        assert np.all(m3.array == np.array([[37, 54], [81, 118]]))
        with pytest.raises(TypeError) as e:
            m4 = m1 @ 2
        assert str(e.value) == "Cannot matrix multiply MyMatrix by int using @. Use * instead."
        with pytest.raises(TypeError) as e:
            m5 = m1 @ [1, 2]
        assert str(e.value) == "Cannot matrix multiply MyMatrix by list using @."
        with pytest.raises(TypeError) as e:
            m3 @= 2
    
    def test_matrixbase___mul__(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m1 = MyMatrix([[1, 2], [3, 4]])
        m2 = m1 * 2
        assert m2.array.shape == (2, 2)
        assert np.all(m2.array == np.array([[2, 4], [6, 8]]))
        m2 *= 2
        assert np.all(m2.array == np.array([[4, 8], [12, 16]]))
        m2 = 0.5 * m2
        assert np.all(m2.array == np.array([[2, 4], [6, 8]]))
        with pytest.raises(TypeError) as e:
            m3 = m1 * [1, 2]
        with pytest.raises(TypeError) as e:
            m4 = m1 * m1
        m5 = 2 * m1
        assert np.all(m5.array == np.array([[2, 4], [6, 8]]))
        with pytest.raises(TypeError) as e:
            m2 *= [3, 1]
    
    def test_matrixbase___truediv__(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        
        m1 = MyMatrix([[2, 4], [6, 8]])
        m2 = m1 / 2
        assert m2.array.shape == (2, 2)
        assert np.all(m2.array == np.array([[1, 2], [3, 4]]))
        m1 /= 2
        assert np.all(m1.array == np.array([[1, 2], [3, 4]]))
        with pytest.raises(NotImplementedError) as e:
            m3 = m1 / [1, 2]
        with pytest.raises(NotImplementedError) as e:
            m4 = m1 / m1
        with pytest.raises(TypeError) as e:
            m5 = 2 / m1
        with pytest.raises(NotImplementedError) as e:
            m1 /= [1, 2]

    def test_matrixbase___eq__(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        m1 = MyMatrix([[1, 2], [3, 4]])
        assert m1 == MyMatrix([[1, 2], [3, 4]])

    def test_matrixbase___neg__(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        m1 = MyMatrix([[1, 2], [3, 4]])
        assert -m1 == MyMatrix([[-1, -2], [-3, -4]])
        assert np.all((-m1).array == np.array([[-1, -2], [-3, -4]]))

    def test_matrixbase___abs__(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        m1 = MyMatrix([[1, 2], [3, 4]])
        assert np.isclose(abs(m1), -2, rtol=MatrixBase.EPSILON, atol=MatrixBase.EPSILON)

    def test_matrixbase_add_sub(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        m1 = MyMatrix([[1, 2], [3, 4]])
        m2 = MyMatrix([[3, 2], [1, 1]])
        assert m2 + m1 == MyMatrix([[4, 4], [4, 5]])
        assert m2 - m1 == MyMatrix([[2, 0], [-2, -3]])

    def test_matrixbase_trace(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        m1 = MyMatrix([[1, 2], [3, 4]])
        m2 = MyMatrix([[3, 2], [1, 1]])
        assert m1.trace == 5
        assert m2.trace == 4

    def test_matrixbase_get_row_col(self):
        class MyMatrix(MatrixBase):
            dim = (2, 2)
            def __init__(self, *args) -> None:
                super().__init__(*args)
        m1 = MyMatrix([[1, 2], [3, 4]])
        assert np.all(m1.get_row(0) == np.array([1, 2]))
        m2 = MyMatrix([[3, 2], [1, 1]])
        assert np.all(m2.get_column(1) == np.array([2, 1]))

