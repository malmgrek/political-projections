"""Generic utilities"""


def throw(ex):
    raise ex


def checklist_to_bool(x):
    return x is not None and "v" in x
