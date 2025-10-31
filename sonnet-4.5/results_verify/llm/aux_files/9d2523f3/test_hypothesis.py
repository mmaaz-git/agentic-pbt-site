import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')
from hypothesis import given, strategies as st, assume
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union
import json


class SharedOptions(BaseModel):
    logit_bias: Optional[Union[dict, str]] = Field(default=None)

    @field_validator("logit_bias")
    def validate_logit_bias(cls, logit_bias):
        if logit_bias is None:
            return None

        if isinstance(logit_bias, str):
            try:
                logit_bias = json.loads(logit_bias)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in logit_bias string")

        validated_logit_bias = {}
        for key, value in logit_bias.items():
            try:
                int_key = int(key)
                int_value = int(value)
                if -100 <= int_value <= 100:
                    validated_logit_bias[int_key] = int_value
                else:
                    raise ValueError("Value must be between -100 and 100")
            except ValueError:
                raise ValueError("Invalid key-value pair in logit_bias dictionary")

        return validated_logit_bias


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(min_size=1), st.integers(), st.booleans(), st.none()),
        min_size=0,
        max_size=5
    )
)
def test_validate_logit_bias_rejects_invalid_keys(invalid_dict):
    assume(any(not k.lstrip('-').isdigit() for k in invalid_dict.keys()))

    try:
        options = SharedOptions(logit_bias=invalid_dict)
        print(f"No error for dict: {invalid_dict}")
        assert False, f"Should have raised ValueError for {invalid_dict}"
    except ValueError:
        pass
    except TypeError as e:
        print(f"TypeError instead of ValueError for dict: {invalid_dict}")
        print(f"  Error: {e}")
        # This is the bug - we get TypeError instead of ValueError

# Run the test
test_validate_logit_bias_rejects_invalid_keys()
print("Test completed")
