import pytest
from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v0.1.0")
def func_one_kwonly(a, *, b=2):
    return a + b


def test_too_many_positional_args():
    with pytest.raises(TypeError):
        func_one_kwonly(1, 2, 3)


if __name__ == "__main__":
    test_too_many_positional_args()