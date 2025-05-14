from typing import Generator

import pytest
import transformers.training_args

# Store the original list to restore it later
ORIGINAL_VALID_DICT_FIELDS = list(
    transformers.training_args._VALID_DICT_FIELDS
)


@pytest.fixture(autouse=True)
def reset_valid_dict_fields() -> Generator[None, None, None]:
    """Reset _VALID_DICT_FIELDS before each test to prevent cross-test contamination.

    Errors occur when using `trl` which has a `SFTConfig` that has a dict-like
    attribute, which mutates `_VALID_DICT_FIELDS` for all other tests.
    """
    # Reset to original state before each test
    transformers.training_args._VALID_DICT_FIELDS = list(
        ORIGINAL_VALID_DICT_FIELDS
    )
    yield
    # Reset again after test just to be safe
    transformers.training_args._VALID_DICT_FIELDS = list(
        ORIGINAL_VALID_DICT_FIELDS
    )
