from enum import Enum


class LabelClasse(Enum):
    Negative = 0
    Neutral = 1
    Positive = 2


def map_label_str_to_class_idx(s: str):
    if s == "مثبت":
        return LabelClasse.Positive.value
    if s == "منفی":
        return LabelClasse.Negative.value
    return LabelClasse.Neutral.value
