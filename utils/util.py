import numpy as np
from math import fabs


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)


def normalize(img, m=10, M=90):
    return np.clip((img - robust_min(img, m)) / (robust_max(img, M) - robust_min(img, m)), 0.0, 1.0)


def first_element_greater_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the minimum value that satisfies values[i] >= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value)
    val = values[i] if i < len(values) else None
    return (i, val)


def last_element_less_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the maximum value that satisfies values[i] <= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value, side='right') - 1
    val = values[i] if i >= 0 else None
    return (i, val)


def closest_element_to(values, req_value):
    """Returns the tuple (i, values[i], diff) such that i is the closest value to req_value,
    and diff = |values(i) - req_value|
    Note: this function assumes that values is a sorted array!"""
    assert(len(values) > 0)

    i = np.searchsorted(values, req_value, side='left')
    if i > 0 and (i == len(values) or fabs(req_value - values[i - 1]) < fabs(req_value - values[i])):
        idx = i - 1
        val = values[i - 1]
    else:
        idx = i
        val = values[i]

    diff = fabs(val - req_value)
    return (idx, val, diff)
