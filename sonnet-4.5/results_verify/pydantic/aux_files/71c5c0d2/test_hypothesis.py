from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel


class Model(BaseModel):
    data: bytes


@given(st.binary(min_size=0, max_size=100))
def test_bytes_field_roundtrip(data):
    m = Model(data=data)
    d = m.dict()
    json_str = m.json()

    recreated_from_dict = Model(**d)
    assert recreated_from_dict.data == data

    recreated_from_json = Model.parse_raw(json_str)
    assert recreated_from_json.data == data

if __name__ == "__main__":
    test_bytes_field_roundtrip()