from hypothesis import given, strategies as st, settings
from pydantic import BaseModel, Field

class ModelWithAliasAndRoundTrip(BaseModel):
    field_one: str = Field(alias="fieldOne")
    field_two: int = Field(alias="fieldTwo")

@st.composite
def alias_roundtrip_strategy(draw):
    field_one = draw(st.text(min_size=1, max_size=50))
    field_two = draw(st.integers(min_value=-1000, max_value=1000))
    return ModelWithAliasAndRoundTrip(**{"fieldOne": field_one, "fieldTwo": field_two})

@settings(max_examples=500)
@given(alias_roundtrip_strategy())
def test_round_trip_without_by_alias(model):
    json_str = model.model_dump_json(by_alias=False, round_trip=True)
    restored = ModelWithAliasAndRoundTrip.model_validate_json(json_str)
    assert model.model_dump() == restored.model_dump()

if __name__ == "__main__":
    test_round_trip_without_by_alias()
    print("Test passed!")