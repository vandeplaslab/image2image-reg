"""Transform file."""

from __future__ import annotations

from pathlib import Path

from koyo.typing import PathLike
from loguru import logger


def transform_elastix(
    files: list[PathLike],
    transform_file: PathLike,
    output_dir: PathLike,
    suffix: str = "_transformed",
    as_uint8: bool | None = None,
    tile_size: int = 512,
    clip: str = "ignore",
    pixel_size: float | None = None,
    inverse: bool = False,
) -> list[Path]:
    """Transform files."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import OmeTiffWriter

    from image2image_reg.elastix.transform import transform_attached_point, transform_attached_shape
    from image2image_reg.elastix.transform_sequence import TransformSequence

    # load transformation
    transform_seq = TransformSequence.from_final(transform_file)
    transform_seq.set_inverse(inverse)

    # transform files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_ = []
    for file in files:
        file = Path(file)
        try:
            reader = get_simple_reader(file, init_pyramid=False, auto_pyramid=False)
        except Exception as e:
            logger.error(e)
            continue

        # export image/points/shapes
        if reader.reader_type == "image":
            writer = OmeTiffWriter(reader, transformer=transform_seq)  # type: ignore[arg-type]
            name = file.stem.replace(".ome", "") + suffix
            path = writer.write(
                name,
                output_dir=output_dir,
                as_uint8=as_uint8,
                tile_size=tile_size,
                overwrite=True,
            )
            files_.append(path)
        elif reader.reader_type == "points":
            if pixel_size is None:
                logger.warning(
                    "You did not specify pixel size, using 1.0 as default. This warning can be ignored if the value"
                    " is actually 1.0."
                )

            path = transform_attached_point(
                transform_seq,
                file,
                pixel_size or 1.0,
                output_dir / (file.stem + suffix + file.suffix),
                silent=False,
                clip=clip,
            )
            files_.append(path)
        elif reader.reader_type == "shapes":
            if pixel_size is None:
                logger.warning(
                    "You did not specify pixel size, using 1.0 as default. This warning can be ignored if the value"
                    " is actually 1.0."
                )
            path = transform_attached_shape(
                transform_seq,
                file,
                pixel_size or 1.0,
                output_dir / (file.stem + suffix + file.suffix),
                silent=False,
                clip=clip,
            )
            files_.append(path)
        else:
            logger.error("Unknown reader type")
    return files_  # type: ignore[return-value]
