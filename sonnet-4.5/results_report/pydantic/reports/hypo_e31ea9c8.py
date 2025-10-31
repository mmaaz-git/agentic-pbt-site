from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

@given(st.text())
@settings(max_examples=200)
def test_str_lower_then_upper(text):
    class Model(BaseModel):
        value: Annotated[str, transform(str.lower).str_upper()]

    m = Model(value=text)
    assert m.value == text.upper(), f"Expected {text.upper()!r} but got {m.value!r}"

test_str_lower_then_upper()