from hypothesis import given, strategies as st
from pandas.io.excel._base import inspect_excel_format
import pytest


def test_inspect_excel_format_empty_raises():
    with pytest.raises(ValueError, match="stream is empty"):
        inspect_excel_format(b'')

if __name__ == "__main__":
    test_inspect_excel_format_empty_raises()
    print("Test passed!")