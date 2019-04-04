import inspect


def list_recursive_concrete_subclasses(base):
    """List all concrete subclasses of `base` recursively."""
    return _filter_concrete(_bfs(base))


def _filter_concrete(classes):
    return list(filter(lambda c: not inspect.isabstract(c), classes))


def _bfs(base):
    return base.__subclasses__() + sum([
        _bfs(subclass)
        for subclass in base.__subclasses__()
    ], [])
