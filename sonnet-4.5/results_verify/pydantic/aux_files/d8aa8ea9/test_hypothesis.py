from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import transform
import pytest


@settings(max_examples=300)
@given(st.one_of(st.integers(), st.text()))
def test_otherwise_with_union_types(value):
    int_pipeline = transform(lambda x: x).ge(0)
    str_pipeline = transform(lambda x: x).str_lower()
    combined_pipeline = int_pipeline.otherwise(str_pipeline)

    class Model(BaseModel):
        x: Annotated[int | str, combined_pipeline]

    try:
        result = Model(x=value)

        if isinstance(value, int) and value >= 0:
            assert result.x == value
        elif isinstance(value, str):
            assert result.x == value.lower()
    except TypeError as e:
        print(f"TypeError on value {repr(value)}: {e}")
        raise

if __name__ == "__main__":
    test_otherwise_with_union_types()