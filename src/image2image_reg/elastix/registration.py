"""Registration model."""

from __future__ import annotations

from enum import Enum, EnumMeta
from pathlib import Path

from koyo.typing import PathLike

from image2image_reg.elastix.registration_map import DEFAULT_REGISTRATION_PARAMETERS_MAP


def _elx_lineparser(
    line: str,
) -> tuple[str, list[str]] | tuple[None, None]:
    if line[0] == "(":
        params = line.replace("(", "").replace(")", "").replace("\n", "").replace('"', "")
        params = params.split(" ", 1)
        k, v = params[0], params[1]
        if " " in v:
            v = v.split(" ")
            v = list(filter(lambda a: a != "", v))
        if not isinstance(v, list):
            v = [v]
        return k, v
    else:
        return None, None


def _read_elastix_parameter_file(
    elx_param_fp: PathLike,
) -> dict[str, list[str]]:
    with open(
        elx_param_fp,
    ) as f:
        lines = f.readlines()
    parameters = {}
    for line in lines:
        k, v = _elx_lineparser(line)
        if k is not None:
            parameters.update({k: v})
    return parameters


class _RegModelMeta(EnumMeta):
    def __getitem__(self, name):
        if isinstance(name, str):
            if "-" in name:
                name = name.replace("-", "_")
        try:
            return super().__getitem__(name)
        except (TypeError, KeyError):
            if isinstance(name, (str, Path)) and Path(name).exists():
                return _read_elastix_parameter_file(name)
            else:
                raise ValueError(
                    f"unrecognized registration parameter ({name}), please provide file path to elastix transform"
                    f" parameters or specify one of {[i.name for i in self]}"
                )


class Registration(dict, Enum, metaclass=_RegModelMeta):
    """
    Default registration parameters. Can also pass a filepath of elastix transforms and these
    will be used.
    """

    rigid: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["rigid"]
    affine: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["affine"]
    similarity: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["similarity"]
    nl: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl"]
    fi_correction: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["fi_correction"]
    nl_reduced: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_reduced"]
    nl_mid: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_mid"]
    nl2: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl2"]
    nl3: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl3"]
    # expanded parameters
    rigid_expanded: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["rigid_expanded"]
    similarity_expanded: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["similarity_expanded"]
    affine_expanded: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["affine_expanded"]
    nl_expanded: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_expanded"]
    # extreme parameters
    rigid_extreme: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["rigid_extreme"]
    similarity_extreme: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["similarity_extreme"]
    affine_extreme: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["affine_extreme"]
    nl_extreme: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_extreme"]
    # advanced mean squares
    rigid_ams: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["rigid_ams"]
    similarity_ams: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["similarity_ams"]
    affine_ams: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["affine_ams"]
    nl_ams: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_ams"]
    # normalized correlation
    rigid_anc: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["rigid_anc"]
    similarity_anc: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["similarity_anc"]
    affine_anc: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["affine_anc"]
    nl_anc: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_anc"]
    # test parameters
    rigid_test: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["rigid_test"]
    affine_test: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["affine_test"]
    similarity_test: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["similarity_test"]
    nl_test: dict[str, list[str]] = DEFAULT_REGISTRATION_PARAMETERS_MAP["nl_test"]

    def __str__(self):
        return self.name

    def __deepcopy__(self, _):
        return self.name

    @classmethod
    def from_name(cls, name: str | Registration) -> Registration:
        """Create a Registration from a name."""
        if isinstance(name, Registration):
            return name
        return getattr(cls, name)
