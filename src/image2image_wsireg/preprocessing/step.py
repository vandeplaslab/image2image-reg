"""Pre-processing step."""


class BaseStep:
    """Pre-processing step."""

    def __init__(self):
        pass

    def __call__(self, image):
        return self.apply(image)

    def to_sitk(self):
        """Convert to SimpleITK filter."""
        raise NotImplementedError

    def to_array(self):
        """Convert to NumPy array."""
        raise NotImplementedError

    def apply(self, image):
        """Apply step to image."""
        raise NotImplementedError
