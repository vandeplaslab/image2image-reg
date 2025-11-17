"""Feature detection unit."""

from __future__ import annotations

import typing as ty

import cv2
import numpy as np
from loguru import logger
from skimage import exposure

logger = logger.bind(src="FeatureDetection")

# maximum number of features in a single image - if the value is larger, it will be filtered down.
N_MAX_FEATURES: int = 20000

# Default feature detector
DEFAULT_FEATURE_DETECTOR = cv2.BRISK_create()  # type: ignore[attr-defined]


def filter_features(kp: list, desc: np.ndarray, n_keep: int = N_MAX_FEATURES) -> tuple[list, np.ndarray]:
    """Get keypoints with the highest response.

    Parameters
    ----------
    kp : list
        List of cv2.KeyPoint detected by an OpenCV feature detector.

    desc : ndarray
        2D numpy array of keypoint descriptors, where each row is a keypoint
        and each column a feature.

    n_keep : int
        Maximum number of features that are retained.

    Returns
    -------
    Keypoints and and corresponding descriptors that the the n_keep highest
    responses.

    """
    response = np.array([x.response for x in kp])
    keep_idx = np.argsort(response)[::-1][0:n_keep]
    return [kp[i] for i in keep_idx], desc[keep_idx, :]


class OpenCVDetector(ty.Protocol):
    """Protocol for OpenCV feature detectors."""

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> list:
        """Detects keypoints in an image."""


class OpenCVDescriptor(ty.Protocol):
    """Protocol for OpenCV feature descriptors."""

    def compute(self, image: np.ndarray, keypoints: list) -> tuple[list, np.ndarray]:
        """Computes the descriptors for the keypoints."""

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[list, np.ndarray]:
        """Detects and computes keypoints in an image."""


class FeatureDetectorBase:
    """Abstract class for feature detection and description.

    User can create other feature detectors as subclasses, but each must
    return keypoint positions in xy coordinates along with the descriptors
    for each keypoint.

    Note that in some cases, such as KAZE, kp_detector can also detect
    features. However, in other cases, there may need to be a separate feature
    detector (like BRISK or ORB) and feature descriptor (like VGG).

    Attributes
    ----------
        kp_detector : object
            Keypoint detector, by default from OpenCV

        kp_descriptor : object
            Keypoint descriptor, by default from OpenCV

        kp_detector_name : str
            Name of keypoint detector

        kp_descriptor : str
            Name of keypoint descriptor

    Methods
    -------
    detectAndCompute(image, mask=None)
        Detects and describes keypoints in image

    """

    def __init__(self, kp_detector: OpenCVDetector | None = None, kp_descriptor: OpenCVDescriptor | None = None):
        """
        Parameters
        ----------
            kp_detector : object
                Keypoint detetor, by default from OpenCV

            kp_descriptor : object
                Keypoint descriptor, by default from OpenCV

        """
        self.kp_detector = kp_detector
        self.kp_descriptor = kp_descriptor

        if kp_descriptor is not None and kp_detector is not None:
            # User provides both a detector and descriptor #
            self.kp_descriptor_name = kp_descriptor.__class__.__name__
            self.kp_detector_name = kp_detector.__class__.__name__

        if kp_descriptor is None and kp_detector is not None:
            # Will be using kp_descriptor for detectAndCompute #
            kp_descriptor = kp_detector
            kp_detector = None

        if kp_descriptor is not None and kp_detector is None:
            # User provides a descriptor, which must also be able to detect #
            self.kp_descriptor_name = kp_descriptor.__class__.__name__
            self.kp_detector_name = self.kp_descriptor_name

            try:
                _img = np.zeros((10, 10), dtype=np.uint8)
                kp_descriptor.detectAndCompute(_img, mask=None)
            except Exception:
                logger.exception(
                    f"{self.kp_descriptor_name} unable to both detect and compute features. "
                    f"Setting to {DEFAULT_FEATURE_DETECTOR.__class__.__name__}"
                )

                self.kp_detector = DEFAULT_FEATURE_DETECTOR

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Detect the features in the image.

        Detect the features in the image using the defined kp_detector, then
        describe the features using the kp_descriptor. The user can override
        this method so they don't have to use OpenCV's Keypoint class.

        Parameters
        ----------
        image : ndarray
            Image in which the features will be detected. Should be a 2D uint8
            image if using OpenCV

        mask : ndarray, optional
            Binary image with same shape as image, where foreground > 0,
            and background = 0. If provided, feature detection  will only be
            performed on the foreground.

        Returns
        -------
        kp : np.ndarray
            (N, 2) array positions of keypoints in xy corrdinates for N
            keypoints

        desc : np.ndarray
            (N, M) array containing M features for each of the N keypoints

        """
        image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        if self.kp_detector is not None:
            detected_kp = self.kp_detector.detect(image)
            kp, desc = self.kp_descriptor.compute(image, detected_kp)

        else:
            kp, desc = self.kp_descriptor.detectAndCompute(image, mask=mask)

        if desc.shape[0] > N_MAX_FEATURES:
            kp, desc = filter_features(kp, desc)
        kp_pos_xy = np.array([k.pt for k in kp])
        return kp_pos_xy, desc


class OrbFD(FeatureDetectorBase):
    """Uses ORB for feature detection and description."""

    def __init__(self, kp_descriptor=cv2.ORB_create(N_MAX_FEATURES)):  # type: ignore[attr-defined]
        super().__init__(kp_descriptor=kp_descriptor)


class BriskFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and description."""

    def __init__(self, kp_descriptor=cv2.BRISK_create()):  # type: ignore[attr-defined]
        super().__init__(kp_descriptor=kp_descriptor)


class KazeFD(FeatureDetectorBase):
    """Uses KAZE for feature detection and description."""

    def __init__(self, kp_descriptor=cv2.KAZE_create(extended=False)):  # type: ignore[attr-defined]
        super().__init__(kp_descriptor=kp_descriptor)


class AkazeFD(FeatureDetectorBase):
    """Uses AKAZE for feature detection and description."""

    def __init__(self, kp_descriptor=cv2.AKAZE_create()):  # type: ignore[attr-defined]
        super().__init__(kp_descriptor=kp_descriptor)


class DaisyFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and DAISY for feature description."""

    def __init__(
        self,
        kp_detector=DEFAULT_FEATURE_DETECTOR,
        kp_descriptor=cv2.xfeatures2d.DAISY_create(),  # type: ignore[attr-defined]
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class LatchFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and LATCH for feature description."""

    def __init__(
        self,
        kp_detector=DEFAULT_FEATURE_DETECTOR,
        kp_descriptor=cv2.xfeatures2d.LATCH_create(rotationInvariance=True),  # type: ignore[attr-defined]
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class BoostFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and Boost for feature description."""

    def __init__(
        self,
        kp_detector=DEFAULT_FEATURE_DETECTOR,
        kp_descriptor=cv2.xfeatures2d.BoostDesc_create(),  # type: ignore[attr-defined]
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class VggFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and VGG for feature description."""

    def __init__(
        self,
        kp_detector=DEFAULT_FEATURE_DETECTOR,
        kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=5.0),  # type: ignore[attr-defined]
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class SensitiveVggFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and VGG for feature description."""

    def __init__(
        self,
        kp_detector=cv2.BRISK_create(thresh=5),  # type: ignore[attr-defined]
        kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=5.0),
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class VerySensitiveVggFD(FeatureDetectorBase):
    """Uses BRISK for feature detection and VGG for feature description."""

    def __init__(
        self,
        kp_detector=cv2.BRISK_create(thresh=2),  # type: ignore[attr-defined]
        kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=5.0),
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class OrbVggFD(FeatureDetectorBase):
    """Uses ORB for feature detection and VGG for feature description."""

    def __init__(
        self,
        kp_detector=cv2.ORB_create(nfeatures=N_MAX_FEATURES, fastThreshold=0),  # type: ignore[attr-defined]
        kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=0.75),  # type: ignore[attr-defined]
    ):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)
