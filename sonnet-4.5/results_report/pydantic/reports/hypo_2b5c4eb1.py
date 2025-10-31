from hypothesis import given, strategies as st, settings, example
from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import validate_as


@given(st.text())
@settings(max_examples=1000)
@example('0\x1f')
@example('\x1f')
def test_str_strip_matches_python_str_strip(text):
    """Property: pipeline.str_strip() should behave identically to Python's str.strip()"""
    pipeline = validate_as(str).str_strip()

    class TestModel(BaseModel):
        field: Annotated[str, pipeline]

    model = TestModel(field=text)
    expected = text.strip()
    actual = model.field

    assert actual == expected, f"str_strip() mismatch: input={text!r}, expected={expected!r}, actual={actual!r}"


if __name__ == "__main__":
    test_str_strip_matches_python_str_strip()