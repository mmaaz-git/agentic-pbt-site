from pydantic.experimental.pipeline import _Pipeline

pipeline = _Pipeline(())

print(f"pipeline.eq(5): {type(pipeline.eq(5))}")
print(f"pipeline == 5: {type(pipeline == 5)}")

print(f"pipeline.not_eq(5): {type(pipeline.not_eq(5))}")
print(f"pipeline != 5: {type(pipeline != 5)}")