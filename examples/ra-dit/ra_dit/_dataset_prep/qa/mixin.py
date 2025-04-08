"""QA Data Prepper"""


class QAMixin:
    @property
    def required_cols(self) -> list[str]:
        return ["answer", "question"]
