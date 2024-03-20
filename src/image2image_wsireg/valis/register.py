"""Registration workflow."""
from __future__ import annotations

import typing as ty
from pathlib import Path

from skimage.transform import SimilarityTransform
from valis.non_rigid_registrars import OpticalFlowWarper
from valis.preprocessing import DEFAULT_COLOR_STD_C, ChannelGetter, ColorfulStandardizer

from image2image_wsireg.valis.detect import VggFD
from image2image_wsireg.valis.matcher import RANSAC_NAME, Matcher

# Default image processing
DEFAULT_BRIGHTFIELD_CLASS = ColorfulStandardizer
DEFAULT_BRIGHTFIELD_PROCESSING_ARGS = {"c": DEFAULT_COLOR_STD_C, "h": 0}
DEFAULT_FLOURESCENCE_CLASS = ChannelGetter
DEFAULT_FLOURESCENCE_PROCESSING_ARGS = {"channel": "dapi", "adaptive_eq": True}
DEFAULT_NORM_METHOD = "img_stats"

# Default rigid registration parameters
DEFAULT_FD = VggFD
DEFAULT_TRANSFORM_CLASS = SimilarityTransform
DEFAULT_MATCH_FILTER = Matcher(match_filter_method=RANSAC_NAME)
DEFAULT_SIMILARITY_METRIC = "n_matches"
DEFAULT_AFFINE_OPTIMIZER_CLASS = None
DEFAULT_MAX_PROCESSED_IMG_SIZE = 850
DEFAULT_MAX_IMG_DIM = 850
DEFAULT_THUMBNAIL_SIZE = 500
DEFAULT_MAX_NON_RIGID_REG_SIZE = 3000

# Default non-rigid registration parameters
DEFAULT_NON_RIGID_CLASS = OpticalFlowWarper
DEFAULT_NON_RIGID_KWARGS: dict = {}

# Default micro-rigid registration parameters
DEFAULT_MICRO_RIGID_CLASS = None
DEFAULT_MICRO_RIGID_KWARGS: dict = {}

# Crop options


class ValisWorkflow:
    """Valis workflow for registration."""

    def __init__(
        self,
        src_dir,
        dst_dir,
        series=None,
        name=None,
        image_type=None,
        feature_detector_cls=DEFAULT_FD,
        transformer_cls=DEFAULT_TRANSFORM_CLASS,
        affine_optimizer_cls=DEFAULT_AFFINE_OPTIMIZER_CLASS,
        similarity_metric=DEFAULT_SIMILARITY_METRIC,
        matcher=DEFAULT_MATCH_FILTER,
        imgs_ordered=False,
        non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
        non_rigid_reg_kws=DEFAULT_NON_RIGID_KWARGS,
        compose_non_rigid=False,
        img_list=None,
        reference_img_f=None,
        align_to_reference=False,
        do_rigid=True,
        crop=None,
        create_masks=True,
        denoise_rigid=True,
        check_for_reflections=False,
        resolution_xyu=None,
        slide_dims_dict_wh=None,
        max_image_dim_px=DEFAULT_MAX_IMG_DIM,
        max_processed_image_dim_px=DEFAULT_MAX_PROCESSED_IMG_SIZE,
        max_non_rigid_registration_dim_px=DEFAULT_MAX_PROCESSED_IMG_SIZE,
        thumbnail_size=DEFAULT_THUMBNAIL_SIZE,
        norm_method=DEFAULT_NORM_METHOD,
        micro_rigid_registrar_cls=DEFAULT_MICRO_RIGID_CLASS,
        micro_rigid_registrar_kws=DEFAULT_MICRO_RIGID_KWARGS,
        qt_emitter=None,
    ):
        """
        src_dir: str
            Path to directory containing the slides that will be registered.

        dst_dir : str
            Path to where the results should be saved.

        name : str, optional
            Descriptive name of registrar, such as the sample's name

        series : int, optional
            Slide series to that was read. If None, series will be set to 0.

        image_type : str, optional
            The type of image, either "brightfield", "fluorescence",
            or "multi". If None, VALIS will guess `image_type`
            of each image, based on the number of channels and datatype.
            Will assume that RGB = "brightfield",
            otherwise `image_type` will be set to "fluorescence".

        feature_detector_cls : FeatureDD, optional
            Uninstantiated FeatureDD object that detects and computes
            image features. Default is VggFD. The
            available feature_detectors are found in the `feature_detectors`
            module. If a desired feature detector is not available,
            one can be created by subclassing `feature_detectors.FeatureDD`.

        transformer_cls : scikit-image Transform class, optional
            Uninstantiated scikit-image transformer used to find
            transformation matrix that will warp each image to the target
            image. Default is SimilarityTransform

        affine_optimizer_cls : AffineOptimzer class, optional
            Uninstantiated AffineOptimzer that will minimize a
            cost function to find the optimal affine transformations.
            If a desired affine optimization is not available,
            one can be created by subclassing `affine_optimizer.AffineOptimizer`.

        similarity_metric : str, optional
            Metric used to calculate similarity between images, which is in
            turn used to build the distance matrix used to sort the images.
            Can be "n_matches", or a string to used as
            distance in spatial.distance.cdist. "n_matches"
            is the number of matching features between image pairs.

        match_filter_method: str, optional
            "GMS" will use filter_matches_gms() to remove poor matches.
            This uses the Grid-based Motion Statistics (GMS) or RANSAC.

        imgs_ordered : bool, optional
            Boolean defining whether or not the order of images in img_dir
            are already in the correct order. If True, then each filename should
            begin with the number that indicates its position in the z-stack. If
            False, then the images will be sorted by ordering a feature distance
            matix. Default is False.

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference.

        align_to_reference : bool, optional
            If `False`, images will be non-rigidly aligned serially towards the
            reference image. If `True`, images will be non-rigidly aligned
            directly to the reference image. If `reference_img_f` is None,
            then the reference image will be the one in the middle of the stack.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images. See
            the `non_rigid_registrars` module for a desciption of available
            methods. If a desired non-rigid registration method is not available,
            one can be implemented by subclassing.NonRigidRegistrar.
            If None, then only rigid registration will be performed

        non_rigid_reg_params: dictionary, optional
            Dictionary containing key, value pairs to be used to initialize
            `non_rigid_registrar_cls`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

        compose_non_rigid : bool, optional
            Whether or not to compose non-rigid transformations. If `True`,
            then an image is non-rigidly warped before aligning to the
            adjacent non-rigidly aligned image. This allows the transformations
            to accumulate, which may bring distant features together but could
            also  result  in un-wanted deformations, particularly around the edges.
            If `False`, the image not warped before being aaligned to the adjacent
            non-rigidly aligned image. This can reduce unwanted deformations, but
            may not bring distant features together.

        img_list : list, dictionary, optional
            List of images to be registered. However, it can also be a dictionary,
            in which case the key: value pairs are full_path_to_image: name_of_image,
            where name_of_image is the key that can be used to access the image from
            Valis.slide_dict.

        do_rigid: bool, dictionary, optional
            Whether or not to perform rigid registration. If `False`, rigid
            registration will be skipped.

            If `do_rigid` is a dictionary, it should contain inverse transformation
            matrices to rigidly align images to the specificed by `reference_img_f`.
            M will be estimated for images that are not in the dictionary.
            Each key is the filename of the image associated with the transformation matrix,
            and value is a dictionary containing the following values:
                `M` : (required) a 3x3 inverse transformation matrix as a numpy array.
                      Found by determining how to align fixed to moving.
                      If `M` was found by determining how to align moving to fixed,
                      then `M` will need to be inverted first.
                `transformation_src_shape_rc` : (optional) shape (row, col) of image used to find the rigid transformation.
                      If not provided, then it is assumed to be the shape of the level 0 slide
                `transformation_dst_shape_rc` : (optional) shape of registered image.
                      If not provided, this is assumed to be the shape of the level 0 reference slide.

        crop: str, optional
            How to crop the registered images. "overlap" will crop to include
            only areas where all images overlapped. "reference" crops to the
            area that overlaps with a reference image, defined by
            `reference_img_f`. This option can be used even if `reference_img_f`
            is `None` because the reference image will be set as the one at the center
            of the stack.

            If both `crop` and `reference_img_f` are `None`, `crop`
            will be set to "overlap". If `crop` is None, but `reference_img_f`
            is defined, then `crop` will be set to "reference".

        create_masks : bool, optional
            Whether or not to create and apply masks for registration.
            Can help focus alignment on the tissue, but can sometimes
            mask too much if there is a lot of variation in the image.

        denoise_rigid : bool, optional
            Whether or not to denoise processed images before rigid registion.
            Note that un-denoised images are used in the non-rigid registration

        check_for_reflections : bool, optional
            Determine if alignments are improved by relfecting/mirroring/flipping
            images. Optional because it requires re-detecting features in each version
            of the images and then re-matching features, and so can be time consuming and
            not always necessary.

        resolution_xyu: tuple, optional
            Physical size per pixel and the unit. If None (the default), these
            values will be determined for each slide using the slides' metadata.
            If provided, this physical pixel sizes will be used for all of the slides.
            This option is available in case one cannot easily access to the original
            slides, but does have the information on pixel's physical units.

        slide_dims_dict_wh : dict, optional
            Key= slide/image file name,
            value= dimensions = [(width, height), (width, height), ...] for each level.
            If None (the default), the slide dimensions will be pulled from the
            slides' metadata. If provided, those values will be overwritten. This
            option is available in case one cannot easily access to the original
            slides, but does have the information on the slide dimensions.

        max_image_dim_px : int, optional
            Maximum width or height of images that will be saved.
            This limit is mostly to keep memory in check.

        max_processed_image_dim_px : int, optional
            Maximum width or height of processed images. An important
            parameter, as it determines the size of of the image in which
            features will be detected and displacement fields computed.

        max_non_rigid_registration_dim_px : int, optional
             Maximum width or height of images used for non-rigid registration.
             Larger values may yeild more accurate results, at the expense of
             speed and memory. There is also a practical limit, as the specified
             size may be too large to fit in memory.

        mask_dict : dictionary
            Dictionary where key = overlap type (all, overlap, or reference), and
            value = (mask, mask_bbox_xywh)

        thumbnail_size : int, optional
            Maximum width or height of thumbnails that show results

        norm_method : str
            Name of method used to normalize the processed images. Options
            are None when normalization is not desired, "histo_match" for
            histogram matching and "img_stats" for normalizing by image statistics.
            See preprocessing.match_histograms and preprocessing.norm_khan
            for details.

        iter_order : list of tuples
            Each element of `iter_order` contains a tuple of stack
            indices. The first value is the index of the moving/current/from
            image, while the second value is the index of the moving/next/to
            image.

        micro_rigid_registrar_cls : MicroRigidRegistrar, optional
            Class used to perform higher resolution rigid registration. If `None`,
            this step is skipped.

        micro_rigid_registrar_params : dictionary
            Dictionary of keyword arguments used intialize the `MicroRigidRegistrar`

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """
        # Get name, based on src directory
        if name is None:
            if src_dir.endswith(os.path.sep):
                name = os.path.split(src_dir[:-1])[1]
            else:
                name = os.path.split(src_dir)[1]
        self.name = name.replace(" ", "_")

        # Set paths #
        self.src_dir = src_dir
        self.dst_dir = os.path.join(dst_dir, self.name)
        self.name_dict = None

        if img_list is not None:
            if isinstance(img_list, dict):
                # Key=original file name, value=name
                self.original_img_list = list(img_list.keys())
                self.name_dict = img_list
            elif hasattr(img_list, "__iter__"):
                self.original_img_list = list(img_list)
            else:
                msg = (
                    f"Cannot upack `img_list`, which is type {type(img_list).__name__}. "
                    "Please provide an iterable object (list, tuple, array, etc...) that has the location of the images"
                )
                valtils.print_warning(msg, rgb=Fore.RED)
        else:
            self.get_imgs_in_dir()

        if self.name_dict is None:
            self.name_dict = self.get_img_names(self.original_img_list)

        self.check_for_duplicated_names(self.original_img_list)

        self.set_dst_paths()

        # Some information may already be provided #
        self.slide_dims_dict_wh = slide_dims_dict_wh
        self.resolution_xyu = resolution_xyu
        self.image_type = image_type

        # Results fields #
        self.series = series
        self.size = 0
        self.aligned_img_shape_rc = None
        self.aligned_slide_shape_rc = None
        self.slide_dict = {}

        # Fields related to image pre-processing #
        self.brightfield_procsseing_fxn_str = None
        self.if_procsseing_fxn_str = None

        if max_image_dim_px < max_processed_image_dim_px:
            msg = f"max_image_dim_px is {max_image_dim_px} but needs to be less or equal to {max_processed_image_dim_px}. Setting max_image_dim_px to {max_processed_image_dim_px}"
            valtils.print_warning(msg)
            max_image_dim_px = max_processed_image_dim_px

        self.max_image_dim_px = max_image_dim_px
        self.max_processed_image_dim_px = max_processed_image_dim_px
        self.max_non_rigid_registration_dim_px = max_non_rigid_registration_dim_px

        # Setup rigid registration #
        self.reference_img_idx = None
        self.reference_img_f = reference_img_f
        self.align_to_reference = align_to_reference
        self.iter_order = None

        self.do_rigid = do_rigid
        self.rigid_registrar = None
        self.micro_rigid_registrar_cls = micro_rigid_registrar_cls
        self.micro_rigid_registrar_params = micro_rigid_registrar_kws
        self.denoise_rigid = denoise_rigid

        self._set_rigid_reg_kwargs(
            name=name,
            feature_detector=feature_detector_cls,
            similarity_metric=similarity_metric,
            matcher=matcher,
            transformer=transformer_cls,
            affine_optimizer=affine_optimizer_cls,
            imgs_ordered=imgs_ordered,
            reference_img_f=reference_img_f,
            check_for_reflections=check_for_reflections,
        )

        # Setup non-rigid registration #
        self.non_rigid_registrar = None
        self.non_rigid_registrar_cls = non_rigid_registrar_cls

        if crop is None:
            if reference_img_f is None:
                self.crop = CROP_OVERLAP
            else:
                self.crop = CROP_REF
        else:
            self.crop = crop

        self.compose_non_rigid = compose_non_rigid
        if non_rigid_registrar_cls is not None:
            self._set_non_rigid_reg_kwargs(
                name=name,
                non_rigid_reg_class=non_rigid_registrar_cls,
                non_rigid_reg_params=non_rigid_reg_kws,
                reference_img_f=reference_img_f,
                compose_non_rigid=compose_non_rigid,
            )

        # Info realted to saving images to view results #
        self.mask_dict = None
        self.create_masks = create_masks

        self.thumbnail_size = thumbnail_size
        self.original_overlap_img = None
        self.rigid_overlap_img = None
        self.non_rigid_overlap_img = None
        self.micro_reg_overlap_img = None

        self.has_rounds = False
        self.norm_method = norm_method
        self.summary_df = None
        self.start_time = None
        self.end_rigid_time = None
        self.end_non_rigid_time = None

        self._empty_slides = {}
