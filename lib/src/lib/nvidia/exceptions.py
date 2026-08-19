class NVIDIAError(Exception):
    """Base error raised by nvidia-lib."""

class ModelCatalogError(NVIDIAError):
    """The models.json catalog is missing or invalid."""

class ModelNotFoundError(NVIDIAError):
    """No model matches the requested identifier."""

class AmbiguousModelError(NVIDIAError):
    """More than one model matches a partial identifier."""

class ModelCredentialError(NVIDIAError):
    """A selected model has no usable endpoint or API key."""
