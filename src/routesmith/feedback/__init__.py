"""Feedback collection and adaptation module."""

from routesmith.feedback.collector import FeedbackCollector, FeedbackRecord
from routesmith.feedback.signals import QualitySignal, SignalExtractor
from routesmith.feedback.storage import FeedbackStorage

__all__ = [
    "FeedbackCollector",
    "FeedbackRecord",
    "FeedbackStorage",
    "QualitySignal",
    "SignalExtractor",
]
