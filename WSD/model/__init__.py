"""Model components for unified WSD."""

from .backbone import EncoderBackbone
from .classifier import Projector, UnifiedConceptClassifier

__all__ = [
    "Projector",
    "EncoderBackbone",
    "UnifiedConceptClassifier",
]
