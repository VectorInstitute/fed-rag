"""QA Data Prepper"""

from ..base_data_prepper import BaseDataPrepper


class QAMixin(BaseDataPrepper):
    @property
    def required_cols(self) -> list[str]:
        return ["answer", "question"]
