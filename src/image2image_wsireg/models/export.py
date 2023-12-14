"""Export model."""
import typing as ty

from pydantic import BaseModel, Field, validator


class Export(BaseModel):
    """Specify how modality should be exported."""

    channel_ids: ty.Optional[list[int]] = Field(None, alias="channel_indices")
    channel_names: ty.Optional[list[str]] = Field(None)
    channel_colors: ty.Optional[list[str]] = Field(None)
    as_uint8: bool = Field(False, alias="export_as_uint8")

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return self.dict(exclude_none=True, exclude_defaults=True)

    @validator("channel_ids", pre=True)
    def _validate_channel_ids(cls, v) -> ty.Optional[list[int]]:
        if v is None:
            return None
        return [int(x) for x in v]

    @validator("channel_names", pre=True)
    def _validate_channel_names(cls, v) -> ty.Optional[list[str]]:
        if v is None:
            return None
        return [str(x) for x in v]

    @validator("channel_colors", pre=True)
    def _validate_channel_colors(cls, v) -> ty.Optional[list[str]]:
        if v is None:
            return None
        return [str(x) for x in v]
