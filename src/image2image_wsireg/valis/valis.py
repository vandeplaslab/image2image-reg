"""Valis registration with some overrides."""
from pathlib import Path

from loguru import logger
from valis.registration import (
    DEFAULT_BRIGHTFIELD_CLASS,
    DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
    DEFAULT_FLOURESCENCE_CLASS,
    DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
)
from valis.registration import Valis as _Valis

from image2image_wsireg.valis.slide_io import Image2ImageSlideReader

logger = logger.bind(src="ValisRegistration")


class Valis(_Valis):
    """Valis registration."""

    def register(
        self,
        brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
        brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
        if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
        if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
        processor_dict=None,
        reader_cls=Image2ImageSlideReader,
        reader_dict=None,
    ):
        """Register a collection of images

        This function will convert the slides to images, pre-process and normalize them, and
        then conduct rigid registration. Non-rigid registration will then be performed if the
        `non_rigid_registrar_cls` argument used to initialize the Valis object was not None.

        In addition to the objects returned, the desination directory (i.e. `dst_dir`)
        will contain thumbnails so that one can visualize the results: converted image
        thumbnails will be in "images/"; processed images in "processed/";
        rigidly aligned images in "rigid_registration/"; non-rigidly aligned images in "non_rigid_registration/";
        non-rigid deformation field images (i.e. warped grids colored by the direction and magntidue)
        of the deformation) will be in ""deformation_fields/". The size of these thumbnails
        is determined by the `thumbnail_size` argument used to initialze this object.

        One can get a sense of how well the registration worked by looking
        in the "overlaps/", which shows how the images overlap before
        registration, after rigid registration, and after non-rigid registration. Each image
        is created by coloring an inverted greyscale version of the processed images, and then
        blending those images.

        The "data/" directory will contain a pickled copy of this registrar, which can be
        later be opened (unpickled) and used to warp slides and/or point data.

        "data/" will also contain the `summary_df` saved as a csv file.


        Parameters
        ----------
        brightfield_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process brightfield images to make
            them look as similar as possible.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process immunofluorescent images
            to make them look as similar as possible.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        processor_dict : dict, optional
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.

        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata. If None (the default),
            the appropriate SlideReader will be found by `slide_io.get_slide_reader`.
            This option is provided in case the slides cannot be opened by a current
            SlideReader class. In this case, the user should create a subclass of
            SlideReader. See slide_io.SlideReader for details.

        reader_dict: dict, optional
            Dictionary specifying which readers to use for individual images. The
            keys should be the image's filename, and the values the instantiated slide_io.SlideReader
            to use to read that file. Valis will try to find an appropritate reader
            for any omitted files, or will use `reader_cls` as the default.

        Returns
        -------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.
            This object can be pickled if so desired

        non_rigid_registrar : SerialNonRigidRegistrar
            SerialNonRigidRegistrar object that performed serial
            non-rigid registration. This object can be pickled if so desired.

        summary_df : Dataframe
            `summary_df` contains various information about the registration.

            The "from" column is the name of the image, while the "to" column
            name of the image it was aligned to. "from" is analagous to "moving"
            or "current", while "to" is analgous to "fixed" or "previous".

            Columns begining with "original" refer to error measurements of the
            unregistered images. Those beginning with "rigid" or "non_rigid" refer
            to measurements related to rigid or non-rigid registration, respectively.

            Columns beginning with "mean" are averages of error measurements. In
            the case of errors based on feature distances (i.e. those ending in "D"),
            the mean is weighted by the number of feature matches between "from" and "to".

            Columns endining in "D" indicate the median distance between matched
            features in "from" and "to".

            Columns ending in "TRE" indicate the target registration error between
            "from" and "to".

            Columns ending in "mattesMI" contain measurements of the Mattes mutual
            information between "from" and "to".

            "processed_img_shape" indicates the shape (row, column) of the processed
            image actually used to conduct the registration

            "shape" is the shape of the slide at full resolution

            "aligned_shape" is the shape of the registered full resolution slide

            "physical_units" are the names of the pixels physcial unit, e.g. u'\u00B5m'

            "resolution" is the physical unit per pixel

            "name" is the name assigned to the Valis instance

            "rigid_time_minutes" is the total number of minutes it took
            to convert the images and then rigidly align them.

            "non_rigid_time_minutes" is the total number of minutes it took
            to convert the images, and then perform rigid -> non-rigid registration.

        """
        from pickle import dump
        from time import time

        self.start_time = time()
        try:
            print("\n==== Converting images\n")
            self.convert_imgs(series=self.series, reader_cls=reader_cls, reader_dict=reader_dict)

            print("\n==== Processing images\n")
            slide_processors = self.create_img_processor_dict(
                brightfield_processing_cls=brightfield_processing_cls,
                brightfield_processing_kwargs=brightfield_processing_kwargs,
                if_processing_cls=if_processing_cls,
                if_processing_kwargs=if_processing_kwargs,
                processor_dict=processor_dict,
            )

            self.brightfield_procsseing_fxn_str = brightfield_processing_cls.__name__
            self.if_processing_fxn_str = if_processing_cls.__name__
            self.process_imgs(processor_dict=slide_processors)

            # print("\n==== Rigid registration\n")
            rigid_registrar = self.rigid_register()
            aligned_slide_shape_rc = self.get_aligned_slide_shape(0)
            self.aligned_slide_shape_rc = aligned_slide_shape_rc
            self.iter_order = rigid_registrar.iter_order
            for slide_obj in self.slide_dict.values():
                slide_obj.aligned_slide_shape_rc = aligned_slide_shape_rc

            if self.micro_rigid_registrar_cls is not None:
                print("\n==== Micro-rigid registration\n")
                self.micro_rigid_register()

            if rigid_registrar is False:
                return None, None, None

            if self.non_rigid_registrar_cls is not None:
                print("\n==== Non-rigid registration\n")
                non_rigid_registrar = self.non_rigid_register(rigid_registrar, slide_processors)

            else:
                non_rigid_registrar = None

            self._add_empty_slides()

            print("\n==== Measuring error\n")
            error_df = self.measure_error()
            self.cleanup()

            data_dir = Path(self.data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
            f_out = self.data_dir / (self.name + "_registrar.pickle")
            self.reg_f = str(f_out)
            with open(f_out, "wb") as f:
                dump(self, f)

            data_f_out = data_dir / (self.name + "_summary.csv")
            error_df.to_csv(data_f_out, index=False)

        except Exception as e:
            logger.exception(e)
            return None, None, None
        return rigid_registrar, non_rigid_registrar, error_df

    def to_cache(self):
        """Save the valis object to the cache."""

    def from_cache(self):
        """Open the valis object from the cache."""
