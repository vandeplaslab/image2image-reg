"""Typing."""

from __future__ import annotations

import typing as ty

if ty.TYPE_CHECKING:
    from image2image_reg.elastix.transform_sequence import TransformSequence
    from image2image_reg.models import Preprocessing


class TransformPair(ty.TypedDict):
    """Transformation pair."""

    registration: TransformSequence | None
    initial: TransformSequence | None


class SourceTargetPair(ty.TypedDict):
    """Registration pair."""

    source: str
    target: str | None


class RegistrationNode(ty.TypedDict):
    """Registration node."""

    modalities: SourceTargetPair
    params: list[str]
    registered: bool
    transforms: TransformPair | None
    transform_tag: str | None
    source_preprocessing: Preprocessing | None
    target_preprocessing: Preprocessing | None


class SerializedRegistrationNode(ty.TypedDict):
    """Serialized registration node."""

    source: str
    target: str
    through: str | None
    reg_params: list[str]
    source_preprocessing: dict[str, ty.Any] | None
    target_preprocessing: dict[str, ty.Any] | None


class SerializedRegisteredRegistrationNode(ty.TypedDict):
    """Serialized registered registration node."""

    modalities: SourceTargetPair
    params: list[str]
    registered: bool
    transform_tag: str
    source_preprocessing: Preprocessing | None
    target_preprocessing: Preprocessing | None


class AttachedImageDict(ty.TypedDict):
    """Attached image dictionary."""

    files: list[str]
    attach_to: str


class AttachedShapeOrPointDict(ty.TypedDict):
    """Attached shape dictionary."""

    files: list[str]
    pixel_size: float
    attach_to: str


class ElastixRegConfig(ty.TypedDict):
    """Configuration."""

    schema_version: str
    name: str
    # output_dir: str
    cache_images: bool
    # cache_dir: str
    pairwise: bool
    registration_paths: dict[str, SerializedRegistrationNode]
    registration_graph_edges: list[SerializedRegisteredRegistrationNode]
    modalities: dict[str, dict]
    attachment_images: dict[str, str]
    attachment_shapes: dict[str, AttachedShapeOrPointDict]
    attachment_points: dict[str, AttachedShapeOrPointDict]
    merge: bool
    merge_images: dict[str, list[str]]


class ValisRegConfig(ty.TypedDict):
    """Configuration."""

    schema_version: str
    name: str
    # output_dir: str
    cache_images: bool
    # cache_dir: str
    modalities: dict[str, dict]
    reference: str | None
    check_for_reflections: bool
    non_rigid_registration: bool
    micro_registration: bool
    micro_registration_fraction: float
    feature_detector: str
    feature_matcher: str
    attachment_images: dict[str, str]
    attachment_shapes: dict[str, AttachedShapeOrPointDict]
    attachment_points: dict[str, AttachedShapeOrPointDict]
    merge: bool
    merge_images: dict[str, list[str]]
