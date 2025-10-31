from hypothesis import given, strategies as st
from pydantic import BaseModel, Field

@given(
    public_value=st.integers(),
    private_value=st.integers()
)
def test_excluded_field_roundtrip(public_value, private_value):
    """
    PROPERTY: model_validate(model_dump(m)) should equal m, even with excluded fields.
    """
    class ExcludeModel(BaseModel):
        public: int
        private: int = Field(default=0, exclude=True)

    model = ExcludeModel(public=public_value, private=private_value)

    dumped = model.model_dump()
    restored = ExcludeModel.model_validate(dumped)

    assert model.private == restored.private, f"Private field mismatch: {model.private} != {restored.private}"
    assert model == restored, f"Models not equal: {model} != {restored}"

if __name__ == "__main__":
    test_excluded_field_roundtrip()