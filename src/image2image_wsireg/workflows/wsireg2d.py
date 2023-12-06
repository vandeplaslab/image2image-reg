"""Image registration in 2D."""
from __future__ import annotations

import time
import typing as ty
from contextlib import suppress
from copy import copy, deepcopy
from pathlib import Path
from warnings import warn

import numpy as np
from image2image_io._reader import is_supported
from koyo.json import write_json_data
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from tqdm import tqdm

from image2image_wsireg.enums import ArrayLike, WriterMode
from image2image_wsireg.models import Modality, Preprocessing, Registration, Transform, TransformSequence
from image2image_wsireg.utils.figures import (
    read_elastix_iteration_dir,
    read_elastix_transform_dir,
    write_iteration_plots,
)
from image2image_wsireg.utils.registration import _prepare_reg_models, register_2d_images, sitk_pmap_to_dict
from image2image_wsireg.wrapper import ImageWrapper


class RegistrationPair(ty.TypedDict):
    """Registration pair."""

    source: str
    target: str | None


class TransformPair(ty.TypedDict):
    """Transformation pair."""

    registration: TransformSequence | None
    initial: list[dict[str, str]] | None


class RegistrationNode(ty.TypedDict):
    """Registration node."""

    modalities: RegistrationPair
    params: list[Registration]
    registered: bool
    transforms: TransformPair | None
    source_preprocessing: Preprocessing | None
    target_preprocessing: Preprocessing | None


class WsiReg2d:
    """Whole slide registration utilizing WsiReg approach of graph based registration."""

    def __init__(self, name: str, output_dir: PathLike, cache_images: bool = True, merge_images: bool = False):
        self.name = name
        self.output_dir = Path(output_dir)
        # setup project directory
        self.project_dir = (self.output_dir / name).with_suffix(".wsireg")
        self.project_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Project directory: {self.project_dir}")
        self.cache_images = cache_images
        self.pairwise = False
        self.merge_images = merge_images

        # setup modalities
        self.modalities: dict[str, Modality] = {}
        self.attachment_images: dict[str, Modality] = {}  # TODO
        self.attachment_shapes: dict[str, Modality] = {}  # TODO
        self.attachment_geojsons: dict[str, Modality] = {}  # TODO
        self.merge_modalities: dict[str, list[str]] = {}  # TODO

        # setup registration paths
        self.registration_paths: dict[str, list[str]] = {}
        self.registration_nodes: list[RegistrationNode] = []
        self.transform_path_map: dict[str, list[RegistrationPair]] = {}
        self.original_size_transforms: dict[str, list[dict]] = {}
        self.transformations: dict[str, dict] = {}

        # cache where we will store temporary data
        self.preprocessed_cache: dict[str, dict] = {"image_spacing": {}, "image_sizes": {}}

        # setup cache directory
        self.cache_dir = self.project_dir / "Cache"
        if self.cache_images:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Image cache directory: {self.cache_dir}")

    def add_modality(
        self,
        name: str,
        path: PathLike,
        pixel_size: float = 1.0,
        channel_names: list[str] | None = None,
        channel_colors: list[str] | None = None,
        preprocessing: Preprocessing | dict[str, ty.Any] | None = None,
        mask: PathLike | np.ndarray | None = None,
        output_pixel_size: tuple[float, float] | None = None,
    ) -> None:
        """Add modality."""
        path = Path(path)
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not is_supported(path):
            raise ValueError("Unsupported file format.")
        if name in self.modalities:
            raise ValueError("Modality name already exists.")
        if isinstance(preprocessing, dict):
            preprocessing = Preprocessing(**preprocessing)
        if isinstance(mask, (str, Path)):
            mask = Path(mask)
            if not mask.exists():
                raise ValueError("Mask path does not exist.")
        if isinstance(output_pixel_size, (int, float)):
            output_pixel_size = (float(output_pixel_size), float(output_pixel_size))

        self.modalities[name] = Modality(
            name=name,
            path=path,
            pixel_size=pixel_size,
            channel_names=channel_names,
            channel_colors=channel_colors,
            preprocessing=preprocessing,
            mask=mask,
            output_pixel_size=output_pixel_size,
        )

    @property
    def n_registrations(self) -> int:
        """Number of registrations to be performed."""
        return len(self.registration_nodes)

    def add_registration_path(
        self,
        source: str,
        target: str,
        through: str | None,
        transform: str | Registration | list[str] | list[Registration],
        preprocessing: dict | None = None,
    ) -> None:
        """Add a registration path between modalities.

        You can define registration from source to target or from source through 'through' to target.
        """
        if source not in self.modalities:
            raise ValueError("Source modality does not exist.")
        if target not in self.modalities:
            raise ValueError("Target modality does not exist.")
        if through is not None:
            if through not in self.modalities:
                raise ValueError("Through modality does not exist.")
            if through == target:
                raise ValueError("Through modality cannot be the same as target.")
            if through == source:
                raise ValueError("Through modality cannot be the same as source.")
        if preprocessing is None:
            preprocessing = {"target": None, "source": None}
        elif isinstance(preprocessing, dict):
            assert "target" in preprocessing, "Preprocessing must contain target key."
            assert "source" in preprocessing, "Preprocessing must contain source key."
        self._add_registration_node(source, target, through, transform, preprocessing)

    def _add_registration_node(
        self,
        source: str,
        target: str,
        through: str | None,
        transform: str | Registration | list[str] | list[Registration],
        preprocessing: dict | None = None,
    ) -> None:
        """Add registration node."""
        # create the registration path
        if through is None:
            self.registration_paths.update({source: [target]})
        else:
            self.registration_paths.update({source: [through, target]})

        # setup override pre-processing
        source_preprocessing, target_preprocessing = None, None
        if preprocessing:
            if preprocessing.get("source"):
                source_preprocessing = Preprocessing(**preprocessing.get("source"))  # type: ignore[arg-type]
            if preprocessing.get("target"):
                target_preprocessing = Preprocessing(**preprocessing.get("target"))  # type: ignore[arg-type]

        # validate transform
        if isinstance(transform, str):
            transform = [Registration.from_name(transform)]
        elif isinstance(transform, Registration):
            transform = [transform]
        elif isinstance(transform, list) and all(isinstance(t, str) for t in transform):
            transform = [Registration.from_name(tr) for tr in transform]

        if not isinstance(transform, list) and all(isinstance(t, Registration) for t in transform):
            raise ValueError("Transform must be a Transform object or a list of Transform objects.")

        # create graph edges
        self.registration_nodes.append(
            {
                "modalities": {"source": source, "target": through},
                "params": transform,
                "registered": False,
                "transforms": None,
                "source_preprocessing": source_preprocessing,
                "target_preprocessing": target_preprocessing,
            }
        )
        self._create_transformation_paths(self.registration_paths)

    def _create_transformation_paths(self, registration_paths: dict[str, list[str]]) -> None:
        """Create the path for each registration."""
        transform_path_map: dict[str, list[RegistrationPair]] = {}
        for key, value in registration_paths.items():
            transform_path_modalities = self.find_registration_path(key, value[-1])
            if not transform_path_modalities:
                raise ValueError(f"Could not find registration path from {key} to {value[-1]}.")
            if self.pairwise:
                transform_path_modalities = transform_path_modalities[:1]
            transform_edges: list[RegistrationPair] = []
            for modality in transform_path_modalities:
                for edge in self.registration_nodes:
                    edge_modality = edge["modalities"]["source"]
                    if modality == edge_modality:
                        transform_edges.append(edge["modalities"])
                    transform_path_map.update({key: transform_edges})
        self.transform_path_map = transform_path_map

    def find_registration_path(self, start: str, end: str, path: list[str] | None = None) -> list[str] | None:
        """Find the path from the 'start' modality to 'end' modality in the graph."""
        if path is None:
            path = []
        path = [*path, start]
        if start == end:
            return path
        if start not in self.registration_paths:
            return None
        for node in self.registration_paths[start]:
            if node not in path:
                extended_path = self.find_registration_path(node, end, path)
                if extended_path:
                    return extended_path
        return None

    def _preprocess_image(self, modality: Modality, preprocessing: Preprocessing | None) -> ImageWrapper:
        """Pre-process images."""
        wrapper = ImageWrapper(modality, preprocessing)
        cached = wrapper.check_cache(self.cache_dir, self.cache_images)
        if not cached:
            wrapper.preprocess()
            wrapper.save_cache(self.cache_dir, self.cache_images)
        else:
            wrapper.load_cache(self.cache_dir, self.cache_images)
        if wrapper.image is None:
            raise ValueError(f"The '{modality.name}' image has not been pre-processed.")

        # update caches
        self.preprocessed_cache["image_spacing"].update({modality.name: wrapper.image.GetSpacing()})
        self.preprocessed_cache["image_sizes"].update({modality.name: wrapper.image.GetSize()})
        return wrapper

    def _generate_registration_graph(self) -> None:
        """Generate registration graph."""

    def _coregister_images(
        self,
        source_wrapper: ImageWrapper,
        target_wrapper: ImageWrapper,
        parameters: list[Registration],
        output_dir: Path,
    ) -> tuple[list[dict], TransformSequence | None]:
        """Co-register images."""
        with MeasureTimer() as timer:
            # co-register images
            transform = register_2d_images(source_wrapper, target_wrapper, _prepare_reg_models(parameters), output_dir)
            # convert transformation to something readable
            transforms = [sitk_pmap_to_dict(tf) for tf in transform]
            # add initial images
            initial_transforms = source_wrapper.initial_transforms
            initial_transforms_seq = None
            if initial_transforms:
                initial_transforms_ = [Transform(t) for t in initial_transforms]
                initial_transforms_index = [idx for idx, _ in enumerate(initial_transforms_)]
                initial_transforms_seq = TransformSequence(initial_transforms_, initial_transforms_index)
            self.original_size_transforms.update({target_wrapper.name: target_wrapper.original_size_transform})
        logger.info(f"Registration took {timer()}.")
        return transforms, initial_transforms_seq

    def _generate_figures(self, source: str, target: str, output_dir: Path) -> None:
        """Generate figures."""
        with MeasureTimer() as timer:
            key = f"{source}_to_{target}"
            transform_data = read_elastix_transform_dir(output_dir)
            iteration_data = read_elastix_iteration_dir(output_dir)

            write_iteration_plots(iteration_data, key, output_dir)
            self.preprocessed_cache["iterations"].update({key: iteration_data})
            self.preprocessed_cache["transformations"].update({key: transform_data})
        logger.info(f"Generated figures for {key} in {timer}.")

    def _collate_transformations(self) -> dict[str, dict]:
        transforms = {}
        edge_modality_pairs = [v["modalities"] for v in self.registration_nodes]
        for modality, edges in self.transform_path_map.items():
            full_tform_seq = TransformSequence()
            for index, edge in enumerate(edges):
                registered_edge_transform: TransformPair = self.registration_nodes[edge_modality_pairs.index(edge)][
                    "transforms"
                ]
                if index == 0:
                    transforms[modality] = {
                        f"initial-{edges[index]['source']}": registered_edge_transform["initial"],
                        f"{str(index).zfill(3)}-to-{edges[index]['target']}": registered_edge_transform["registration"],
                    }
                    if registered_edge_transform["initial"]:
                        full_tform_seq.append(registered_edge_transform["initial"])
                    full_tform_seq.append(registered_edge_transform["registration"])
                else:
                    transforms[modality][
                        f"{str(index).zfill(3)}-to-{edges[index]['target']}"
                    ] = registered_edge_transform["registration"]
                    full_tform_seq.append(registered_edge_transform["registration"])
                transforms[modality]["full-transform-seq"] = full_tform_seq
        return transforms

    def run(self) -> None:
        """Execute workflow."""
        # export to configuration file
        self.save_to_wsireg(registered=False)

        # compute transformation information
        for edge in tqdm(self.registration_nodes):
            if edge["registered"]:
                continue

            # retrieve modalities
            source = edge["modalities"]["source"]
            source_modality = deepcopy(self.modalities[source])
            source_preprocessing = edge["source_preprocessing"]
            target = edge["modalities"]["target"]
            target_modality = deepcopy(self.modalities[target])
            target_preprocessing = edge["target_preprocessing"]

            source_wrapper = self._preprocess_image(source_modality, source_preprocessing)
            target_wrapper = self._preprocess_image(target_modality, target_preprocessing)

            # create registration directory
            registration_dir = self.project_dir / f"{source}-{target}_reg_output"
            registration_dir.mkdir(exist_ok=True, parents=True)

            # register images
            registration, initial = self._coregister_images(
                source_wrapper, target_wrapper, edge["params"], registration_dir
            )
            edge["transforms"] = {"registration": registration, "initial": initial}
            edge["registered"] = True

            # load plot data
            self._generate_figures(source, target, registration_dir)

        # collect transformations
        self.transformations = self._collate_transformations()
        # save transformations
        self.save_transformations()
        # write images
        self.write_images()

    @staticmethod
    def _transforms_to_txt(transformations: dict[str, TransformSequence]) -> dict[str, list[str]]:
        tform_txt = {}
        for key, transform in transformations.items():
            if key == "full-transform-seq":
                continue
            if "initial" in key:
                if transform:
                    for idx, registered_transform in enumerate(transform.transforms):
                        tform_txt.update({f"{key}-{str(idx).zfill(2)}": registered_transform.elastix_transform})
            else:
                tform_txt.update({key: [rt.elastix_transform for rt in transform.transforms]})
        return tform_txt

    def _find_not_registered_modalities(self) -> list[str]:
        registered_modalities = [edge["modalities"]["source"] for edge in self.registration_nodes]
        non_reg_modalities = list(set(self.modalities.keys()).difference(registered_modalities))

        # remove attachment modalities
        for attachment_modality in self.attachment_images.keys():
            non_reg_modalities.pop(non_reg_modalities.index(attachment_modality))
        return non_reg_modalities

    def _check_if_all_registered(self) -> bool:
        if not all(reg_edge.get("registered") for reg_edge in self.registration_nodes):
            warn("registration has not been executed for the graph " "no transformations to save")
            return False
        return True

    def save_transformations(self) -> list[Path] | None:
        """Save all transformations for a given modality as JSON."""
        if not self._check_if_all_registered():
            return None

        out = []
        for modality in self.registration_paths:
            final_modality = self.registration_paths[modality][-1]

            output_path = self.output_dir / f"{self.name}-{modality}_to_{final_modality}_transformations.json"
            tform_txt = self._transforms_to_txt(self.transformations[modality])
            write_json_data(output_path, tform_txt)
            out.append(output_path)
            logger.trace(f"Saved transformations to {output_path} for {modality}.")
        # for modality, attachment_modality in self.attachment_images.items():
        #     if attachment_modality not in self._find_nonreg_modalities():
        #         final_modality = self.registration_paths[attachment_modality][-1]
        #         output_path = self.output_dir / f"{self.name}-{modality}_to_{final_modality}_transformations.json"
        #         tform_txt = self._transforms_to_txt(self.transformations[modality])
        #         write_json_data(output_path, tform_txt)
        return out

    def write_images(
        self,
        writer: WriterMode = "ome-tiff",
        write_not_registered: bool = True,
        remove_merged: bool = True,
        to_original_size: bool = True,
    ) -> list | None:
        """Export images after applying transformation."""
        if not self._check_if_all_registered():
            return None

        paths = []
        # prepare merge modalities metadata
        merge_modalities = []
        if len(self.merge_modalities.keys()) > 0:
            for _k, v in self.merge_modalities.items():
                merge_modalities.extend(v)

        modalities = list(self.registration_paths.keys())
        non_registered_modalities = self._find_not_registered_modalities()

        if remove_merged:
            for merge_modality in merge_modalities:
                with suppress(ValueError):
                    index = modalities.index(merge_modality)
                    modalities.pop(index)
                with suppress(ValueError):
                    index = non_registered_modalities.index(merge_modality)
                    non_registered_modalities.pop(index)

        # export modalities
        for modality in tqdm(modalities, desc="Exporting registered modalities..."):
            (im_data, transformations, output_path) = self._prepare_reg_image_transform(
                modality, attachment=False, to_original_size=to_original_size
            )

            path = self._transform_write_image(im_data, transformations, output_path, writer=writer)
            paths.append(path)

        # export attachment modalities
        for modality, attachment_modality in tqdm(
            self.attachment_images.items(), desc="Exporting attachment modalities..."
        ):
            if modality in merge_modalities and remove_merged:
                continue
            if attachment_modality in self._find_not_registered_modalities():
                im_data, transformations, output_path = self._prepare_nonreg_image_transform(
                    modality,
                    attachment=True,
                    attachment_modality=attachment_modality,
                    to_original_size=to_original_size,
                )
            else:
                im_data, transformations, output_path = self._prepare_reg_image_transform(
                    modality,
                    attachment=True,
                    attachment_modality=attachment_modality,
                    to_original_size=to_original_size,
                )
            path = self._transform_write_image(im_data, transformations, output_path, writer=writer)
            paths.append(path)

        # export merge modalities
        if len(self.merge_modalities.items()) > 0:
            path = self._transform_write_merge_images(to_original_size=to_original_size)
            paths.append(path)

        # export non-registered nodes
        if write_not_registered:
            # preprocess and save unregistered nodes
            for modality in tqdm(non_registered_modalities, desc="Exporting not-registered images..."):
                if modality in merge_modalities and remove_merged:
                    continue

                im_data, transformations, output_path = self._prepare_nonreg_image_transform(
                    modality,
                    to_original_size=to_original_size,
                )
                path = self._transform_write_image(im_data, transformations, output_path, writer=writer)
                paths.append(path)
        return path

    def _prepare_reg_image_transform(
        self,
        edge_key: str,
        attachment: bool = False,
        attachment_modality: str | None = None,
        to_original_size: bool = True,
    ) -> tuple[Modality, TransformSequence, Path]:
        if attachment and attachment_modality:
            final_modality = self.registration_paths[attachment_modality][-1]
            transformations = copy(self.transformations[attachment_modality]["full-transform-seq"])
        else:
            final_modality = self.registration_paths[edge_key][-1]
            transformations = copy(self.transformations[edge_key]["full-transform-seq"])

        output_path = self.output_dir / f"{self.name}-{edge_key}_to_{final_modality}_registered"
        im_data_key = None
        if attachment and attachment_modality:
            im_data_key = copy(edge_key)
            edge_key = attachment_modality

        im_data = self.modalities[edge_key]
        if self.original_size_transforms.get(final_modality) and to_original_size:
            logger.trace("Adding transform to original size...")
            original_size_transform = self.original_size_transforms[final_modality]
            if isinstance(original_size_transform, list):
                original_size_transform = original_size_transform[0]
            orig_size_rt = TransformSequence(Transform(original_size_transform), transform_sequence_index=[0])
            transformations.append(orig_size_rt)

        if im_data.preprocessing and im_data.preprocessing.downsample > 1:
            if not im_data.output_pixel_size:
                output_spacing_target = self.modalities[final_modality].output_pixel_size
                transformations.set_output_spacing((output_spacing_target, output_spacing_target))
            else:
                transformations.set_output_spacing(im_data.output_pixel_size)
        elif im_data.output_pixel_size:
            transformations.set_output_spacing(im_data.output_pixel_size)
        if attachment and im_data_key:
            im_data = self.modalities[im_data_key]
        return im_data, transformations, output_path

    def _transform_write_image(
        self, modality, transformations: TransformSequence | None, filename: Path, writer: WriterMode = "ome.tiff"
    ):
        """Transform and write image."""
        from image2image_io.writers import OmeTiffWriter

        wrapper = ImageWrapper(modality.path)

        if writer in ["ome-tiff", "ome-tiff-by-plane"]:
            writer = OmeTiffWriter(wrapper.reader, transformer=transformations)
        else:
            raise ValueError("Other writers are nto yet supported")

        path = writer.write(filename.stem, output_dir=self.output_dir)
        return path

    def save_to_wsireg(self, filename: PathLike | None = None, registered: bool = False) -> Path:
        """Save workflow configuration."""
        import yaml  # type: ignore[import]

        ts = time.strftime("%Y%m%d-%H%M%S")
        status = "registered" if registered is True else "setup"

        registration_paths = {}
        for index, edge in enumerate(self.registration_nodes):
            source = edge["modalities"].get("source")
            if len(self.registration_paths[source]) > 1:
                through = self.registration_paths[source][0]
            else:
                through = None
            target = self.registration_paths[source][-1]
            registration_paths.update(
                {
                    f"reg_path_{index}": {
                        "src_modality_name": edge["modalities"].get("source"),
                        "tgt_modality_name": target,
                        "thru_modality": through,
                        "reg_params": edge.get("params"),
                    }
                }
            )

        # clean-up edges
        reg_graph_edges = deepcopy(self.registration_nodes)
        [rge.pop("transforms", None) for rge in reg_graph_edges]

        modalities_out: dict[str, dict] = {}
        for modality in self.modalities.values():
            modalities_out[modality.name] = modality.dict(exclude_none=True, exclude_defaults=True)
            modalities_out[modality.name].pop("path")
            modalities_out[modality.name]["image_filepath"] = str(modality.path)
            if isinstance(modality.path, ArrayLike):
                modalities_out[modality.name]["image_filepath"] = "ArrayLike"
            if isinstance(modality.preprocessing, Preprocessing):
                modalities_out[modality.name]["preprocessing"] = deepcopy(
                    modality.preprocessing.dict(exclude_none=True, exclude_defaults=True)
                )

        # write config
        config = {
            "project_name": self.name,
            "output_dir": str(self.output_dir),
            "cache_images": self.cache_images,
            "modalities": modalities_out,
            "registration_paths": registration_paths,
            "reg_graph_edges": reg_graph_edges if status == "registered" else None,
            "original_size_transforms": self.original_size_transforms if status == "registered" else None,
            "attachment_shapes": self.attachment_geojsons if len(self.attachment_geojsons) > 0 else None,
            "attachment_images": self.attachment_images if len(self.attachment_images) > 0 else None,
            "merge": self.merge_images,
            "merge_modalities": self.merge_modalities if len(self.merge_modalities) > 0 else None,
        }

        if not filename:
            filename = self.output_dir / f"{ts}-{self.name}-configuration-{status}.yaml"
        filename = Path(filename)
        with open(str(filename), "w") as f:
            yaml.dump(config, f, sort_keys=False)
        return filename
