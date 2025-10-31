import datetime
from typing import Annotated
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

@given(st.datetimes(timezones=st.timezones()))
def test_datetime_tz_should_work(dt):
    pipeline = validate_as(datetime.datetime).datetime_tz(dt.tzinfo)

    class TestModel(BaseModel):
        field: Annotated[datetime.datetime, pipeline]

    model = TestModel(field=dt)
    assert model.field.tzinfo == dt.tzinfo

test_datetime_tz_should_work()