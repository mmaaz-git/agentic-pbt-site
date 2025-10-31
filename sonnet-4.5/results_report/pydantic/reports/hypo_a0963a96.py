from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

@given(st.text())
@settings(max_examples=200)
def test_str_strip_matches_python(text):
    class Model(BaseModel):
        value: Annotated[str, validate_as(str).str_strip()]

    m = Model(value=text)
    assert m.value == text.strip()

if __name__ == "__main__":
    test_str_strip_matches_python()