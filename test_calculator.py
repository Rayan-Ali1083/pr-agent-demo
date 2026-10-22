import pytest

from calculator import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(4, 3) == 12


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero_raises_value_error():
    # Fails now: divide() raises ZeroDivisionError, not ValueError
    with pytest.raises(ValueError):
        divide(10, 0)
