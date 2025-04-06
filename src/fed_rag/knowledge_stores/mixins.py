import uuid

from pydantic import BaseModel, Field


class ManagedMixin(BaseModel):
    ks_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
