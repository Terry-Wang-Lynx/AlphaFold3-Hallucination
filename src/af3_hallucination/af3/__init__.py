"""Optional AlphaFold 3/JAX implementation.

Import this subpackage only inside an environment that provides JAX and the
official AlphaFold 3 source tree. The top-level package has no such dependency.
"""

from .config import ContactSpec, DesignConfig, LossWeights, RematConfig, StageConfig

__all__ = ["ContactSpec", "DesignConfig", "LossWeights", "RematConfig", "StageConfig"]
