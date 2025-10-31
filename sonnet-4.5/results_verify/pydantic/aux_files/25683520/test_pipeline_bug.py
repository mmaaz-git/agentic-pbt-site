from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import _Pipeline


@given(st.integers())
def test_eq_operator_returns_pipeline_not_bool(x):
    pipeline = _Pipeline(())

    method_result = pipeline.eq(x)
    operator_result = pipeline == x

    assert isinstance(method_result, _Pipeline), \
        f"pipeline.eq({x}) should return _Pipeline, got {type(method_result)}"

    assert isinstance(operator_result, _Pipeline), \
        f"pipeline == {x} should return _Pipeline, got {type(operator_result)}"


@given(st.integers())
def test_ne_operator_returns_pipeline_not_bool(x):
    pipeline = _Pipeline(())

    method_result = pipeline.not_eq(x)
    operator_result = pipeline != x

    assert isinstance(method_result, _Pipeline), \
        f"pipeline.not_eq({x}) should return _Pipeline, got {type(method_result)}"

    assert isinstance(operator_result, _Pipeline), \
        f"pipeline != {x} should return _Pipeline, got {type(operator_result)}"


if __name__ == "__main__":
    test_eq_operator_returns_pipeline_not_bool()
    test_ne_operator_returns_pipeline_not_bool()