"""Image class for storing image info and rigid registration parameters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from koyo.typing import PathLike


class Image:
    """Class store info about an image, including the rigid registration parameters.

    Attributes
    ----------
    image : ndarray
        Greyscale image that will be used for feature detection. This images
        should be greyscale and may need to have undergone preprocessing to
        make them look as similar as possible.

    path : str
        full path to the image

    index : int
        ID of the image, based on its ordering `processed_src_dir`

    name : str
        Name of the image. Usually `img_f` but with the extension removed.

    desc : ndarray
        (N, M) array of N descriptors for each keypoint, each of which has
        M features

    kp_pos_xy : ndarray
        (N, 2) array of position for each keypoint

    match_dict : dict
        Dictionary of image matches. Key= img_obj this ZImage is being
        compared to, value= MatchInfo containing information about the
        comparison, such as the position of matches, features for each match,
        number of matches, etc... The MatchInfo objects in this dictionary
        contain only the info for matches that were considered "good".

    unfiltered_match_dict : dict
        Dictionary of image matches. Key= img_obj this ZImage is being
        compared to, value= MatchInfo containing information about the
        comparison, such as the position of matches, features for each match,
        number of matches, etc... The MatchInfo objects in this dictionary
        contain info for all matches that were cross-checked.

    stack_idx : int
        Position of image in sorted Z-stack

    fixed_obj : Image
        ZImage to which this ZImage was aligned, i.e. this is the "moving"
        image, and `fixed_obj` is the "fixed" image. This is set during
        the `align_to_prev` method of the SerialRigidRegistrar. The
        `fixed_obj` will either be immediately above or immediately
        below this ZImage in the image stack.

    reflection_M : ndarray
        Transformation to reflect the image in the x and/or y axis, before padding.
        Will be the first transformation performed

    T : ndarray
        Transformation matrix that translates the image such that it is in a
        padded image that has the same shape as all other images

    to_prev_A : ndarray
        Transformation matrix that warps image to align with the previous image

    optimal_M : ndarray
        Transformation matrix found by minimizing a cost function.
        Used as final optional step to refine alignment

    crop_T : ndarray
        Transformation matrix used to crop image after registration

    M : ndarray
        Final transformation matrix that aligns image in the Z-stack.

    M_inv : ndarray
        Inverse of final transformation matrix that aligns image in
        the Z-stack.

    registered_img : ndarray
        image after being warped

    padded_shape_rc : tuple
        Shape of padded image. All other images will have this shape

    registered_shape_rc = tuple:
        Shape of aligned image. All other aligned images will have this shape

    """

    def __init__(self, image: np.ndarray, path: PathLike, index: int, name: str):
        """Class that stores information about an image.

        Parameters
        ----------
        image : ndarray
            Greyscale image that will be used for feature detection. This
            images should be single channel uint8 images, and may need to
            have undergone preprocessing and/or normalization to make them
            look as similar as possible.

        path : str
            full path to `image`

        index : int
            ID of the image, based on its ordering in the image source directory

        name : str
            Name of the image. Usually img_f but with the extension removed.

        """
        self.image = image
        self.path = Path(path)
        self.index = index
        self.name = name

        self.desc = None
        self.kp_pos_xy = None
        self.match_dict = {}
        self.unfiltered_match_dict = {}
        self.stack_idx = None
        self.fixed_obj = None

        self.reflection_M = np.identity(3)
        self.T = np.identity(3)
        self.to_prev_A = np.identity(3)
        self.optimal_M = np.identity(3)
        self.crop_T = np.identity(3)
        self.M = np.identity(3)
        self.M_inv = np.identity(3)
        self.registered_img = None
        self.padded_shape_rc: tuple[int, int] | None = None
        self.registered_shape_rc: tuple[int, int] | None = None

    def reduce(self, prev_img_obj: Image, next_img_obj: Image) -> None:
        """Reduce amount of info stored, which can take up a lot of space.

        No longer need all descriptors. Only keep match info for neighbors

        Parameters
        ----------
        prev_img_obj : Image
            Instance of image before this Image

        next_img_obj :  Image
            Instance of image after this Image
        """
        self.desc = None
        for img_obj in self.match_dict.keys():
            if prev_img_obj is not None and next_img_obj is not None:
                if prev_img_obj != img_obj and img_obj != next_img_obj:
                    # In middle of stack
                    self.match_dict[img_obj] = None

                elif prev_img_obj is None and img_obj != next_img_obj:
                    # First image doesn't have a previous neighbor
                    self.match_dict[img_obj] = None

                elif prev_img_obj != img_obj and next_img_obj is None:
                    # Last image doesn't have a next neighbor
                    self.match_dict[img_obj] = None
