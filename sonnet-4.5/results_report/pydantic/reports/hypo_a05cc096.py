from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError
from pydantic.types import Base64Bytes

@given(st.binary(min_size=1))
def test_base64bytes_no_silent_data_loss(data):
    class Model(BaseModel):
        field: Base64Bytes

    try:
        m = Model(field=data)
        assert len(m.field) > 0 or len(data) == 0, \
            f"Silent data loss: {len(data)} bytes became {len(m.field)} bytes"
    except ValidationError:
        pass

if __name__ == "__main__":
    test_base64bytes_no_silent_data_loss()