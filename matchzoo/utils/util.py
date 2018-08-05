"""Utility functions."""


class dotdict(dict):
    """make dict have the dict.attr access mode."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
