from hypothesis import given, strategies as st
from pandas.util._validators import _check_for_default_values
import pytest


class RaisesTypeErrorOnEq:
    """Object that raises TypeError when compared with =="""
    def __eq__(self, other):
        raise TypeError("Cannot compare")


@given(st.integers())
def test_check_for_default_values_crashes_on_typeerror(value):
    """
    Property: When comparison raises non-ValueError exceptions,
    the function should fall back to 'is' comparison, not crash.
    """
    obj = RaisesTypeErrorOnEq()
    arg_val_dict = {'param': obj}
    compat_args = {'param': obj}

    with pytest.raises(TypeError):
        _check_for_default_values('test_func', arg_val_dict, compat_args)


if __name__ == "__main__":
    test_check_for_default_values_crashes_on_typeerror()