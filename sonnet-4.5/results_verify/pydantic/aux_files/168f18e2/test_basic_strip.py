from pydantic import BaseModel, ConfigDict
from typing import Annotated
from pydantic_core import core_schema as cs

class SimpleStripModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    field: str

# Test basic whitespace stripping
test1 = SimpleStripModel(field=' hello ')
print(f"Test 1: Input=' hello ' -> Result={test1.field!r}")

test2 = SimpleStripModel(field='\thello\t')
print(f"Test 2: Input='\\thello\\t' -> Result={test2.field!r}")

test3 = SimpleStripModel(field='hello\x1f')
print(f"Test 3: Input='hello\\x1f' -> Result={test3.field!r}")

# Now try with the pipeline
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(str).str_strip()

class PipelineStripModel(BaseModel):
    field: Annotated[str, pipeline]

ptest1 = PipelineStripModel(field=' hello ')
print(f"\nPipeline Test 1: Input=' hello ' -> Result={ptest1.field!r}")

ptest2 = PipelineStripModel(field='\thello\t')
print(f"Pipeline Test 2: Input='\\thello\\t' -> Result={ptest2.field!r}")

ptest3 = PipelineStripModel(field='hello\x1f')
print(f"Pipeline Test 3: Input='hello\\x1f' -> Result={ptest3.field!r}")