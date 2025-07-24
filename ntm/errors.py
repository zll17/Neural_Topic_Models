class NTMError(Exception):
    """Base error type for ntm package."""


class UnsupportedModelError(NTMError):
    """Raised when requested model is not supported."""


class CheckpointFormatError(NTMError):
    """Raised when checkpoint format cannot be parsed."""
