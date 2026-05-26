"""Portable filesystem path serialization helpers."""

from __future__ import annotations

import hashlib
import shutil
import typing as ty
from pathlib import Path, PurePath
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from koyo.typing import PathLike
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PathRootMap = dict[str, str]
SerializedPath = dict[str, str]
PathValue = str | SerializedPath


class PortablePathError(ValueError):
    """Raised when a portable path cannot be resolved."""


class MissingPathRootError(PortablePathError):
    """Raised when a serialized path references an unknown root."""

    def __init__(self, root_name: str | None):
        super().__init__(f"Path root '{root_name}' is not configured.")


class UnknownPathTypeError(PortablePathError):
    """Raised when a serialized path uses an unknown type."""

    def __init__(self, path_type: str | None):
        super().__init__(f"Unknown serialized path type: {path_type!r}")


class ProjectDirectoryRequiredError(PortablePathError):
    """Raised when a project-relative path is resolved without a project directory."""

    def __init__(self, path: SerializedPath):
        super().__init__(f"Cannot resolve relative path without project directory: {path!r}")


class OptionalPathDependencyError(ImportError):
    """Raised when an optional path backend is required but unavailable."""

    def __init__(self):
        super().__init__(
            "Resolving non-file URI paths requires optional dependency 'fsspec'. "
            "Install image2image-reg with the 'paths' extra.",
        )


class PathSettings(BaseSettings):
    """Settings for machine-specific path roots."""

    model_config = SettingsConfigDict(env_prefix="I2REG_", env_nested_delimiter="__")

    path_roots: dict[str, str] = Field(default_factory=dict)


def load_path_roots(path_roots: ty.Mapping[str, PathLike] | None = None) -> PathRootMap:
    """Load path root mappings from settings and explicit overrides."""
    roots = {name: _normalize_path_text(root) for name, root in PathSettings().path_roots.items()}
    if path_roots:
        roots.update({name: _normalize_path_text(root) for name, root in path_roots.items()})
    return roots


def serialize_path(
    path: PathLike | PurePath | SerializedPath,
    project_dir: PathLike | None = None,
    path_roots: ty.Mapping[str, PathLike] | None = None,
) -> str | SerializedPath:
    """Serialize a path as a portable config value.

    Parameters
    ----------
    path
        Local path, URI, or previously serialized path value.
    project_dir
        Project directory used as the base for project-relative paths.
    path_roots
        Named local roots. If ``path`` is under one of these roots, the
        serialized value stores the root name and relative path.
    """
    if isinstance(path, dict):
        if "path_type" not in path:
            raise UnknownPathTypeError(path.get("path_type"))
        return path

    path_text = _normalize_path_text(path)
    if not path_text:
        return path_text
    if _is_uri(path_text):
        return {"path_type": "uri", "uri": path_text}

    if project_dir is None and not path_roots:
        return path_text

    if project_dir is not None:
        relative_path = _relative_to_base(path_text, project_dir)
        if relative_path is not None:
            return {"path_type": "relative", "base": "project", "path": relative_path}

    sorted_roots = sorted(load_path_roots(path_roots).items(), key=lambda item: len(item[1]), reverse=True)
    for root_name, root_path in sorted_roots:
        relative_path = _relative_to_base(path_text, root_path)
        if relative_path is not None:
            return {"path_type": "root", "root": root_name, "path": relative_path}

    if _is_absolute_path_text(path_text):
        return {"path_type": "absolute", "path": path_text}
    if project_dir is not None:
        return {"path_type": "relative", "base": "project", "path": path_text}
    return path_text


def is_serialized_path(path: ty.Any) -> ty.TypeGuard[SerializedPath]:
    """Return whether a config value already uses portable path serialization."""
    return isinstance(path, dict) and "path_type" in path


def upgrade_path(
    path: PathLike | PurePath | SerializedPath | None,
    project_dir: PathLike | None = None,
    path_roots: ty.Mapping[str, PathLike] | None = None,
) -> PathValue | None:
    """Upgrade a legacy config path value to the portable format.

    Already serialized path dictionaries are preserved unchanged. Empty values
    and the historical ``ArrayLike`` sentinel are also preserved because they do
    not represent filesystem paths.
    """
    if path is None:
        return None
    if is_serialized_path(path):
        return path
    if isinstance(path, dict):
        raise UnknownPathTypeError(path.get("path_type"))
    path_text = _normalize_path_text(path)
    if not path_text or path_text == "ArrayLike":
        return path_text
    return serialize_path(path_text, project_dir=project_dir, path_roots=path_roots)


def upgrade_path_list(
    paths: ty.Iterable[PathLike | PurePath | SerializedPath],
    project_dir: PathLike | None = None,
    path_roots: ty.Mapping[str, PathLike] | None = None,
) -> list[PathValue]:
    """Upgrade a list of legacy config path values to the portable format."""
    upgraded: list[PathValue] = []
    for path in paths:
        value = upgrade_path(path, project_dir=project_dir, path_roots=path_roots)
        if value is not None:
            upgraded.append(value)
    return upgraded


def resolve_path(
    path: PathLike | PurePath | SerializedPath,
    project_dir: PathLike | None = None,
    path_roots: ty.Mapping[str, PathLike] | None = None,
    cache_dir: PathLike | None = None,
) -> Path | str:
    """Resolve a serialized path into a local path or protocol string."""
    if not isinstance(path, dict):
        path_text = _normalize_path_text(path)
        if _is_uri(path_text):
            return _resolve_uri(path_text, cache_dir=cache_dir)
        if project_dir is not None and not _is_absolute_path_text(path_text):
            return Path(project_dir) / path_text
        return Path(path_text)

    path_type = path.get("path_type")
    if path_type == "relative":
        base = path.get("base")
        if base != "project" or project_dir is None:
            raise ProjectDirectoryRequiredError(path)
        return Path(project_dir) / path["path"]
    if path_type == "root":
        root_name = path.get("root")
        roots = load_path_roots(path_roots)
        if not root_name or root_name not in roots:
            raise MissingPathRootError(root_name)
        return Path(roots[root_name]) / path["path"]
    if path_type == "absolute":
        return Path(path["path"])
    if path_type == "uri":
        return _resolve_uri(path["uri"], cache_dir=cache_dir)
    raise UnknownPathTypeError(path_type)


def serialize_path_list(
    paths: ty.Iterable[PathLike | PurePath | SerializedPath],
    project_dir: PathLike | None = None,
    path_roots: ty.Mapping[str, PathLike] | None = None,
) -> list[str | SerializedPath]:
    """Serialize a list of paths using the same base and root mappings."""
    return [serialize_path(path, project_dir=project_dir, path_roots=path_roots) for path in paths]


def resolve_path_list(
    paths: ty.Iterable[PathLike | PurePath | SerializedPath],
    project_dir: PathLike | None = None,
    path_roots: ty.Mapping[str, PathLike] | None = None,
    cache_dir: PathLike | None = None,
) -> list[Path | str]:
    """Resolve a list of serialized path values."""
    return [resolve_path(path, project_dir=project_dir, path_roots=path_roots, cache_dir=cache_dir) for path in paths]


def _resolve_uri(uri: str, cache_dir: PathLike | None = None) -> Path | str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = url2pathname(unquote(parsed.path))
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        return Path(path)
    if cache_dir is None:
        return uri
    return _copy_uri_to_cache(uri, cache_dir)


def _copy_uri_to_cache(uri: str, cache_dir: PathLike) -> Path:
    try:
        import fsspec
    except ImportError as error:
        raise OptionalPathDependencyError from error

    parsed = urlparse(uri)
    filename = Path(parsed.path).name or "remote-file"
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
    output_dir = Path(cache_dir) / "RemotePaths"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{Path(filename).stem}-{digest}{Path(filename).suffix}"
    if output_path.exists():
        return output_path
    with fsspec.open(uri, "rb") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return output_path


def _relative_to_base(path: str, base: PathLike) -> str | None:
    relative_path = _relative_to_local_base(path, base)
    if relative_path is not None:
        return relative_path
    return _relative_to_text_base(path, _normalize_path_text(base))


def _relative_to_local_base(path: str, base: PathLike) -> str | None:
    if _looks_like_windows_path(path) or _is_uri(path):
        return None
    path_obj = Path(path).expanduser()
    base_obj = Path(base).expanduser()
    try:
        relative_path = path_obj.absolute().relative_to(base_obj.absolute())
    except ValueError:
        return None
    return relative_path.as_posix()


def _relative_to_text_base(path: str, base: str) -> str | None:
    path_text = _normalize_path_text(path).rstrip("/")
    base_text = _normalize_path_text(base).rstrip("/")
    if not path_text or not base_text:
        return None
    case_sensitive = not (_looks_like_windows_path(path_text) or _looks_like_windows_path(base_text))
    path_compare = path_text if case_sensitive else path_text.casefold()
    base_compare = base_text if case_sensitive else base_text.casefold()
    if path_compare == base_compare:
        return "."
    prefix = f"{base_compare}/"
    if path_compare.startswith(prefix):
        return path_text[len(base_text) + 1 :]
    return None


def _normalize_path_text(path: PathLike | PurePath) -> str:
    if isinstance(path, PurePath):
        return path.as_posix()
    return str(path).replace("\\", "/")


def _is_uri(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme and len(parsed.scheme) > 1)


def _looks_like_windows_path(path: str) -> bool:
    return (len(path) >= 3 and path[1] == ":" and path[2] == "/") or path.startswith("//")


def _is_absolute_path_text(path: str) -> bool:
    return _looks_like_windows_path(path) or Path(path).is_absolute()
