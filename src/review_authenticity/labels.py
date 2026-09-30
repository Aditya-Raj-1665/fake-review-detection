"""One canonical mapping shared by training, metrics, and serving."""

from enum import IntEnum


class Label(IntEnum):
    """Encode review classes consistently."""

    DECEPTIVE = 0
    GENUINE = 1


LABEL_NAMES = {Label.DECEPTIVE: "deceptive", Label.GENUINE: "genuine"}
LABEL_VALUES = {name: int(value) for value, name in LABEL_NAMES.items()}
LABEL_ORDER = [int(Label.DECEPTIVE), int(Label.GENUINE)]


def predicted_label(genuine_probability: float, threshold: float) -> Label:
    """Apply the configured genuine-class decision threshold."""
    return Label.GENUINE if genuine_probability >= threshold else Label.DECEPTIVE
