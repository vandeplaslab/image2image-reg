"""Export model."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class Export(BaseModel):
    """Specify how modality should be exported."""

    as_uint8: bool = False
    channel_ids: list[int] | None = None
    channel_names: list[str] | None = None
    channel_colors: list[str] | None = None

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return self.dict(exclude_none=True, exclude_defaults=True)

    @field_validator("channel_ids", mode="before")
    @classmethod
    def _validate_channel_ids(cls, v) -> list[int] | None:
        if v is None:
            return None
        return [int(x) for x in v]

    @field_validator("channel_names", mode="before")
    @classmethod
    def _validate_channel_names(cls, v) -> list[str] | None:
        if v is None:
            return None
        return [str(x) for x in v]

    @field_validator("channel_colors", mode="before")
    @classmethod
    def _validate_channel_colors(cls, v) -> list[str] | None:
        if v is None:
            return None
        return [str(x) for x in v]
