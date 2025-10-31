import datetime
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(datetime.datetime).datetime_tz(datetime.timezone.utc)

class TestModel(BaseModel):
    field: Annotated[datetime.datetime, pipeline]

dt = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
model = TestModel(field=dt)