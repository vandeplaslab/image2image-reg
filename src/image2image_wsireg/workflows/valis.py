"""Valis interface."""

from pathlib import Path

from koyo.typing import PathLike
from koyo.utilities import is_installed

if not is_installed("valis"):
    raise ImportError("Please install valis to use this module.")


def valis_registration(image_dir: PathLike, reference: PathLike, name: str, output_dir: PathLike) -> None:
    import numpy as np
    import valis
    from image2image_io.readers import ArrayImageReader, get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array
    from koyo.timer import MeasureTimer
    from natsort import natsorted
    from valis import feature_detectors, registration, slide_tools, valtils

    from image2image_wsireg.valis.detect import SensitiveVggFD
    from image2image_wsireg.valis.preprocessing import MaxIntensityProjection

    base_dir = Path(r".").resolve()
    image_dir = base_dir / "pre-registered"

    ims_preaf = reference = image_dir / r"ims-preaf_to_postaf_registered.ome.tiff"
    ims_preaf = reference = image_dir / r"ims-postaf_registered.ome.tiff"
    # codex_preaf = image_dir / r"codex-precodex_registered.ome.tiff"
    codex_preaf = image_dir / r"codex-codex_to_precodex_registered.ome.tiff"

    filelist = [codex_preaf, ims_preaf]
    print(filelist)
    for path in filelist:
        assert path.exists(), f"{path} does not exist."

    channel_kws = {
        valtils.get_name(str(codex_preaf)): [
            MaxIntensityProjection,
            {
                "channel_names": ["DAPI", "MUC1", "CD57", "CD7", "CD90", "CD66", "MUC2"],
                "adaptive_eq": True,
            },
        ],
        valtils.get_name(str(ims_preaf)): [MaxIntensityProjection, {"channel_names": ["EGFP"], "adaptive_eq": True}],
    }

    # initialize java
    valis.registration.init_jvm()

    # ranges
    short_filelist = filelist
    short_filelist = [str(s) for s in short_filelist]
    print(short_filelist)

    name = "valis-codex-to-ims7"
    registered_dir = base_dir / name / "registered"
    registered_dir.mkdir(exist_ok=True, parents=True)

    print(name)
    try:
        # Perform high resolution rigid registration using the MicroRigidRegistrar
        registrar = registration.Valis(
            str(base_dir),
            str(base_dir),
            name=name,
            image_type="fluorescence",
            imgs_ordered=True,
            img_list=short_filelist,
            reference_img_f=str(reference),
            align_to_reference=True,
            check_for_reflections=True,
            feature_detector_cls=SensitiveVggFD,
        )

        with MeasureTimer() as timer:
            rigid_registrar, non_rigid_registrar, error_df = registrar.register(
                # if_processing_cls=MaxIntensityProjection,
                # if_processing_kwargs={"adaptive_eq": True},
                processor_dict=channel_kws
            )
        print(f"Registered low-res images in {timer()}")

        # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
        micro_reg_fraction = 0.5  # Fraction full resolution used for non-rigid registration
        try:
            img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
            min_max_size = np.min([np.max(d) for d in img_dims])
            img_areas = [np.multiply(*d) for d in img_dims]
            max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
            micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)
        except Exception:
            micro_reg_size = 5000  # 5k pixels

        print(f"Micro-registering using {micro_reg_size} pixels.")
        # Perform high resolution non-rigid registration
        with MeasureTimer() as timer:
            try:
                micro_reg, micro_error = registrar.register_micro(
                    max_non_rigid_registration_dim_px=micro_reg_size,
                    processor_dict=channel_kws,
                    reference_img_f=str(reference),
                    align_to_reference=True,
                )
            except Exception as exc:
                print(f"Error during non-rigid registration: {exc}")
        print(f"Registered high-res images in {timer()}")

        # We can also plot the high resolution matches using `Valis.draw_matches`:
        try:
            matches_dst_dir = Path(registrar.dst_dir) / "matches"
            registrar.draw_matches(matches_dst_dir)
        except Exception:
            print("Failed to export matches.")

        # export images to OME-TIFFs
        slide_ref = registrar.get_ref_slide()
        for slide_obj in registrar.slide_dict.values():
            reader = get_simple_reader(slide_obj.src_f)
            output_filename = registered_dir / reader.path.name
            if output_filename.exists():
                continue

            warped = slide_obj.warp_slide(level=0, interp_method="nearest", crop="reference")
            if not isinstance(warped, np.ndarray):
                warped = slide_tools.vips2numpy(warped)

            # ensure that RGB remains RGB but AF remain AF
            if warped.ndim == 3 and np.argmin(warped.shape) == 2:
                warped = np.moveaxis(warped, 2, 0)

            write_ome_tiff_from_array(
                output_filename,
                None,
                warped,
                resolution=slide_ref.resolution,
                channel_names=reader.channel_names,
            )

    except Exception as exc:
        registration.kill_jvm()
        raise exc
    registration.kill_jvm()
