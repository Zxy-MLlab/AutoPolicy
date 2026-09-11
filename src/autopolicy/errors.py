class AutoPolicyError(RuntimeError):
    """Base error with a user-actionable message."""


class ConfigError(AutoPolicyError):
    """Raised when configuration violates the pipeline contract."""


class StageError(AutoPolicyError):
    """Raised when a pipeline stage cannot produce a valid artifact."""


class SafetyError(AutoPolicyError):
    """Raised when deployment safety gates are not satisfied."""
