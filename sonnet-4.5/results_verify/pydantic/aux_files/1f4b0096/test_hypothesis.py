from hypothesis import given, strategies as st, settings
import pydantic.v1 as pv1
from pydantic.v1 import BaseModel, Field


@given(st.text())
@settings(max_examples=500)
def test_exclude_defaults_in_dict(required):
    class ModelWithDefaults(BaseModel):
        required: str
        with_default: int = 42
        with_factory: list = Field(default_factory=list)

    model = ModelWithDefaults(required=required)
    d = model.dict(exclude_defaults=True)

    assert 'required' in d
    assert 'with_default' not in d, "with_default should be excluded when exclude_defaults=True"
    assert 'with_factory' not in d, "with_factory should be excluded when exclude_defaults=True"

if __name__ == "__main__":
    test_exclude_defaults_in_dict()