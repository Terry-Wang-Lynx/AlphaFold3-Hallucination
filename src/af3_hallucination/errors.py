"""Public exception hierarchy."""


class AF3HallucinationError(Exception):
    """Base class for user-facing project errors."""


class ConfigurationError(AF3HallucinationError, ValueError):
    """Raised when a configuration violates the public schema."""


class PluginError(AF3HallucinationError, RuntimeError):
    """Raised when plugin discovery or execution fails."""


class ResumeError(AF3HallucinationError, RuntimeError):
    """Raised when an existing run cannot be resumed safely."""


class OptionalDependencyError(AF3HallucinationError, ImportError):
    """Raised when a requested backend is not installed."""
